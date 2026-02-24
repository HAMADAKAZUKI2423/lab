from __future__ import annotations
import torch
from torch import nn
from .pyramidFunctionModel import PyramidFunctionModel
from .contrastExtraction import ContrastExtraction

class VisFunctionModel(PyramidFunctionModel, ContrastExtraction):
    def __init__(self, level: int,
                dims: int,
                corr_ksize: int,
                col_conversion: str,
                no_mask: bool,
                device: torch.device):
        PyramidFunctionModel.__init__(self, level, dims, device)
        ContrastExtraction.__init__(self, corr_ksize, device)
        
        self.col_conversion = col_conversion
        self.no_mask = no_mask
        # responseに対してmaskを適用しない
        # MLP modelにおいて，何もないところでの反応を抑制するよう学習してほしいため
        self.ignore_residual = False
        self.std_vector: nn.Parameter | None = None
        
    # std_vector explicit
    def compute_std(self, img: torch.Tensor, info_pyr: list[dict[str]], show_plot=False) -> list[dict[str]]:
        with torch.no_grad():
            yuvimg = self.convert_color_v1(img, self.col_conversion)
            pyr = self.gen_Lpyr(yuvimg, self.level, get_gpyr=False)
            # lev=self.lev_cont-1
            # gpyr = self.gen_Gpyr(yuvimg, self.level+lev)
            # col_sigma=self.col_sigma
            
            for i in range(self.level):
                contrast = pyr[i]
                # if i==self.level-1:
                #     contrast = pyr[i]
                # else:
                #     blend_low = gpyr[i+lev]
                #     for j in range(lev):
                #         blend_low = self.upsample(blend_low)
                #     denom = torch.abs(blend_low) + col_sigma.view(1,-1,1,1)
                #     contrast = pyr[i] / denom
                
                info_pyr[i]['std'] += (contrast**2).mean(dim=(2,3)).sum(dim=0) #[B,C]
                info_pyr[i]['count']+=contrast.shape[0]
        return info_pyr
    
    def set_std_vector(self, info_pyr:list[dict[str]]):
        std_vector = []
        for i in range(self.level):
            std_vector.append(info_pyr[i]['std'])
        self.std_vector = nn.Parameter(torch.stack(std_vector,dim=0))
        self.std_vector.requires_grad=False
    
    def showParams(self, required_grad_only=True):
        for name, param in self.named_parameters():
            if required_grad_only:
                if param.requires_grad:
                    print(name)
            else:
                print(name, param.data)
        return
    
    def compute_masked_response(self, resp, numband:int | None = None, no_abs=False):
        if numband == None:
            numband = self.level
            if self.ignore_residual:
                numband -= 1
        total = []

        for i in range(numband):
            if i==self.level-1:
                tmp_resp = resp[i]
            else:
                # mask out irrelevant region
                if self.no_mask:
                    tmp_resp = resp[i]
                else:
                    tmp_resp = resp[i] * self.dilated_mask_gp[i]
            # if self.low_res_visibility:
            #     abs_resp = torch.abs(tmp_resp)
            #     for rep in range(self.level-i-1):
            #         abs_resp = self.downsample(abs_resp)
            #     spat_mean = abs_resp

            # else:
            for rep in range(i):
                tmp_resp = self.upsample(tmp_resp)
            if no_abs:
                spat_mean = tmp_resp
            else:
                spat_mean = torch.abs(tmp_resp)
            total.append(spat_mean)
        total = torch.cat(total, dim=1)
        return total