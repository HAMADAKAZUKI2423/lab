from __future__ import annotations
import torch
from torch.nn import functional as F
from .visFunctionModel import VisFunctionModel

class VisModel(VisFunctionModel):
    def __init__(self, level: int,
                dims: int,
                corr_ksize: int,
                col_conversion: str,
                no_mask: bool,
                target_type: str,
                ovl_input: str,
                device: torch.device):
        super(VisModel, self).__init__(level, dims, corr_ksize, col_conversion, no_mask, device)
        self.reset_state()
        self._set_target_type(target_type)
        self._set_ovl_input(ovl_input)
    
    def _set_target_type(self, target_type: str):
        assert target_type in ["background", "content"]
        self.target_type = target_type

    def _set_ovl_input(self, ovl_input: str):
        assert ovl_input in ["lowfreq", "overlaid"]
        self.ovl_input = ovl_input

    def reset_state(self):
        # image infos
        self.blend: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.ref: torch.Tensor | None = None
        self.blend_img: torch.Tensor | None = None
        self.target_img: torch.Tensor | None = None
        self.ref_img: torch.Tensor | None = None
        self.blend_pyr: list[torch.Tensor] | None = None
        self.target_pyr: list[torch.Tensor] | None = None
        self.ref_pyr: list[torch.Tensor] | None = None
        self.alphamap: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.dilated_mask_gp: list[torch.Tensor] | None = None

        self.raw_ovl: torch.Tensor | None = None

        # process infos
        self.extracted_pyr: list[torch.Tensor] | None = None
        self.extracted_resp: torch.Tensor | None = None
        self.weight_pyr: list[torch.Tensor] | None = None
        self.weight_resp: torch.Tensor | None = None
        self.weight_map: torch.Tensor | None = None
        self.weigheted_resp: torch.Tensor | None = None
        self.aggregated_resp: torch.Tensor | None = None
        self.vis_map: torch.Tensor | None = None
        self.vis_score: float = None
        self.norm_vismap: torch.Tensor | None = None
        self.norm_score: float | None = None

        # for bezier sigmoid
        self.max_vis_score = None
        self.max_vis_map = None
        self.raw_score = None
    
    def set_inputs_bg_ovl_blended(self, bg: torch.Tensor, ovl: torch.Tensor, blend: torch.Tensor, mask: torch.Tensor):
        self.reset_state()
        self.set_background(bg)
        self.set_overlaid(ovl)
        self.set_blend(blend)
        self.set_mask(mask)
        self.raw_ovl = ovl
    
    def set_inputs_tg_ref_blended(self, tg: torch.Tensor, ref: torch.Tensor, blend: torch.Tensor, mask: torch.Tensor):
        self.reset_state()
        self.set_target(tg)
        self.set_reference(ref)
        self.set_blend(blend)
        self.set_mask(mask)
        if self.target_type == "content":
            self.raw_ovl = tg
        else:
            self.raw_ovl = ref

    # alphamap inputs can not use for semi-transparent content (ovl)
    def set_inputs_bg_ovl_alphamap(self, bg: torch.Tensor, ovl: torch.Tensor, alphamap: torch.Tensor, mask: torch.Tensor, blend_mode: str='linear'):
        self.reset_state()
        self.set_background(bg)
        self.set_overlaid(ovl)
        self.set_alphamap(alphamap, blend_mode)
        self.set_mask(mask)
        self.raw_ovl = ovl
    
    def set_inputs_tg_ref_alphamap(self, tg: torch.Tensor, ref: torch.Tensor, alphamap: torch.Tensor, mask: torch.Tensor, blend_mode: str='linear'):
        self.reset_state()
        self.set_target(tg)
        self.set_reference(ref)
        self.set_alphamap(alphamap, blend_mode)
        self.set_mask(mask)
        if self.target_type == "content":
            self.raw_ovl = tg
        else:
            self.raw_ovl = ref
    
    def set_inputs_bg_ovl_contents_blended(self, bg: torch.Tensor, ovl: torch.Tensor, content_color: torch.Tensor, content_alpha: torch.Tensor, blend: torch.Tensor, mask: torch.Tensor):
        self.set_inputs_bg_ovl_contents(bg,ovl,content_color,content_alpha,mask)
        self.set_blend(blend)

    def set_inputs_bg_ovl_contents(self, bg: torch.Tensor, ovl: torch.Tensor, content_color: torch.Tensor, content_alpha: torch.Tensor, mask: torch.Tensor):
        # this does not inputs blend image
        self.reset_state()
        self.set_background(bg)
        ovl_lowfreq = content_color * content_alpha + self.calc_lowfreq_img(bg) * (1 - content_alpha)
        if self.ovl_input == "lowfreq":
            self.set_overlaid(ovl_lowfreq)
        elif self.ovl_input == "overlaid":
            self.set_overlaid(ovl)
        self.set_mask(mask)
        self.raw_ovl = ovl
    
    def set_blend(self, blend: torch.Tensor):
        self.blend = blend
        self.blend_img = self.convert_color_v1(blend, self.col_conversion)
        self.blend_pyr = self.gen_Lpyr(self.blend_img, self.level)
    def set_background(self, bg: torch.Tensor):
        if self.target_type == "background":
            self.set_target(bg)
        if self.target_type == "content":
            self.set_reference(bg)
    def get_background(self) -> torch.Tensor | None:
        if self.target_type == "background":
            return self.target
        if self.target_type == "content":
            return self.ref
    def set_overlaid(self, ovl: torch.Tensor): # background of overlaid must be low-pass filtered
        if self.target_type == "background":
            self.set_reference(ovl)
        if self.target_type == "content":
            self.set_target(ovl)
    def get_overlaid(self) -> torch.Tensor | None:
        # This May returns lowfreq-bg ovl image, not raw bg.
        # ie. ovl as inputs to vismodel
        if self.target_type == "background":
            return self.ref
        if self.target_type == "content":
            return self.target
    def get_raw_overlaid(self) -> torch.Tensor | None:
        # ovl as input image
        return self.raw_ovl
    
    def set_target(self, target: torch.Tensor):
        self.target = target
        self.target_img = self.convert_color_v1(target, self.col_conversion)
        self.target_pyr = self.gen_Lpyr(self.target_img, self.level)
    def set_reference(self, ref: torch.Tensor):
        self.ref = ref
        self.ref_img = self.convert_color_v1(ref, self.col_conversion)
        self.ref_pyr = self.gen_Lpyr(self.ref_img, self.level)
    
    def set_alphamap(self, alphamap: torch.Tensor, blend_mode: str='linear'):
        assert self.target != None and self.ref != None
        self.alphamap = alphamap
        if self.target_type == "background":
            blend = self.blending(self.ref, self.target, alphamap, blend_mode)
        elif self.target_type == "content":
            blend = self.blending(self.target, self.ref, alphamap, blend_mode)
        self.set_blend(blend)
        
    def set_mask(self, mask: torch.Tensor):
        self.mask = mask
        self.generate_maskPyr(mask)

    def compute_visibility(self):
        raise NotImplementedError()
    
    def compute_weights(self):
        raise NotImplementedError()
    
    def compute_visibility_wo_weight(self):
        raise NotImplementedError()

    def projection(self):
        raise NotImplementedError()
    
    def visualize_weights(self):
        raise NotImplementedError()
    
    def visibility_to_norm(self, vis: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def visibility_to_rawscale(self, vis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_params(self):
        raise NotImplementedError()
    
    def save_img(self, out_dir: str, save_only_image: bool):
        raise NotImplementedError()
    
    