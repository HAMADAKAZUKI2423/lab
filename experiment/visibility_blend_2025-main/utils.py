from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torchvision.transforms as T
import sys
import mpl_toolkits.axes_grid1
import torch.nn.functional as F

class stimulus():
    def __init__(self):
        self.index: list[str] | None = None
        self.bg: torch.Tensor | None = None
        self.content_color: torch.Tensor | None = None
        self.content_alpha: torch.Tensor | None = None
        self.ovl: torch.Tensor | None = None
        self.blend: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.ovl_black: torch.Tensor | None = None
        self.vismap: torch.Tensor | None = None

    def set_index(self, index: list[str]):
        self.index = index

    def set_bg(self, bg: torch.Tensor):
        self.bg = bg

    def set_content_color(self, content_color: torch.Tensor):
        self.content_color = content_color

    def set_content_alpha(self, content_alpha: torch.Tensor):
        self.content_alpha = content_alpha

    def set_ovl(self, ovl: torch.Tensor):
        self.ovl = ovl

    def set_blend(self, blend: torch.Tensor):
        self.blend = blend
    
    def set_mask(self, mask: torch.Tensor):
        self.mask = mask
    
    def set_ovl_black(self, ovl_black: torch.Tensor):
        self.ovl_black = ovl_black
    
    def set_vismap(self, vismap: torch.Tensor):
        self.vismap = vismap
    
    def set_save_dir(self, save_dir: str):
        self.save_dir = save_dir
    
    def set_save_name(self, save_name: str):
        self.save_name = save_name

        
    def center_crop(self, img: torch.Tensor, crop_size: int) -> torch.Tensor:
        """
        Crop the input tensor at the center to the specified size.

        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).
            crop_size (int): Desired crop size (crop_size x crop_size).

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        _, _, h, w = img.shape
        # top = (h - crop_size) // 2
        top = (h - crop_size)
        left = (w - crop_size) // 2
        return img[:, :, top:top + crop_size, left:left + crop_size]

    def resize_image(self, img: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Resize the input tensor by a scaling factor.

        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).
            scale (float): Scaling factor.

        Returns:
            torch.Tensor: Resized image tensor.
        """
        from torch.nn.functional import interpolate
        _, _, h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        return interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    def set_from_imgdict(self, imgdict: dict, path: str, device: str):
        assert imgdict.get('bg', 'none') != 'none'
        self.bg = load_img_torch(path + imgdict['bg'], device)

        crop_size = imgdict.get('crop', None)
        resize_scale = imgdict.get('resize', None)

        if resize_scale is not None:
            resize_scale = float(resize_scale)
            self.bg = self.resize_image(self.bg, resize_scale)

        if crop_size is not None:
            crop_size = int(crop_size)
            self.bg = self.center_crop(self.bg, crop_size)

        self.ovl_black: torch.Tensor | None = None
        if imgdict.get('content', 'none') != 'none':
            content = load_img_torch(path + imgdict['content'], device, "alpha")
            if resize_scale is not None:
                content = self.resize_image(content, resize_scale)
            if crop_size is not None:
                content = self.center_crop(content, crop_size)
            self.content_color = content[:, :3]
            self.content_alpha = content[:, 3:]
            self.ovl_black = self.content_color * self.content_alpha
            self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
        elif imgdict.get('opaque', 'none') != 'none':
            self.content_color = load_img_torch(path + imgdict['opaque'], device)
            if resize_scale is not None:
                self.content_color = self.resize_image(self.content_color, resize_scale)
            if crop_size is not None:
                self.content_color = self.center_crop(self.content_color, crop_size)
            self.content_alpha = torch.ones_like(self.bg[:, :1, :, :], device=device, dtype=torch.float32)
            self.ovl_black = self.content_color * self.content_alpha
            self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
        elif imgdict.get('content-opaque', 'none') != 'none':
            content = load_img_torch(path + imgdict['content-opaque'], device, "alpha")
            if resize_scale is not None:
                content = self.resize_image(content, resize_scale)
            if crop_size is not None:
                content = self.center_crop(content, crop_size)
            self.content_color = content[:, :3]
            self.content_alpha = content[:, 3:]
            self.ovl_black = self.content_color * self.content_alpha
            self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
            self.content_color = self.ovl
            self.content_alpha = torch.ones_like(self.content_alpha, device=device, dtype=torch.float32)
        else:
            sys.exit("Error: input must have content or opaque image path")

        if imgdict.get('blend', 'none') != 'none':
            self.blend = load_img_torch(path + imgdict['blend'], device)
            if resize_scale is not None:
                self.blend = self.resize_image(self.blend, resize_scale)
            if crop_size is not None:
                self.blend = self.center_crop(self.blend, crop_size)

        if imgdict.get('mask', 'none') == 'none':
            if self.content_alpha is None:
                self.mask = torch.ones_like(self.bg[:, :1, :, :], device=device, dtype=torch.float32)
            else:
                self.mask = torch.where(self.content_alpha > 0, 1., 0.).to(torch.float32)
        elif imgdict.get('mask', 'none') == 'full':
            self.mask = torch.ones_like(self.bg[:, :1, :, :], device=device, dtype=torch.float32)
        else:
            self.mask = load_img_torch(path + imgdict['mask'], device, "gray")
            if resize_scale is not None:
                self.mask = self.resize_image(self.mask, resize_scale)
            if crop_size is not None:
                self.mask = self.center_crop(self.mask, crop_size)

        if imgdict.get('vismap', 'none') != 'none':
            if imgdict.get('vismap', 'none') != 'mask':
                self.vismap = load_img_torch(path + imgdict['vismap'], device, "gray")
                if resize_scale is not None:
                    self.vismap = self.resize_image(self.vismap, resize_scale)
                if crop_size is not None:
                    self.vismap = self.center_crop(self.vismap, crop_size)
            else:
                self.vismap = self.mask * imgdict.get('vis_value', 'none')
        else:
            if imgdict.get('vis_value', 'none') != 'none':
                assert 0. <= imgdict.get('vis_value', 'none') <= 1.0
                self.vismap = torch.ones_like(self.bg[:, :1, :, :], device=device, dtype=torch.float32) * imgdict.get('vis_value', 'none')

    # def set_from_imgdict(self, imgdict: dict, path: str, device: str):
    #     assert imgdict.get('bg','none') != 'none'
    #     self.bg = load_img_torch(path + imgdict['bg'], device)

    #     self.ovl_black: torch.Tensor | None = None
    #     if imgdict.get('content','none') != 'none':
    #         content = load_img_torch(path + imgdict['content'], device, "alpha")
    #         self.content_color = content[:,:3]
    #         self.content_alpha = content[:,3:]
    #         self.ovl_black = self.content_color * self.content_alpha
    #         self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
    #     elif imgdict.get('opaque','none') != 'none':
    #         self.content_color = load_img_torch(path + imgdict['opaque'], device)
    #         self.content_alpha = torch.ones_like(self.bg[:,:1,:,:], device=device, dtype = torch.float32)
    #         self.ovl_black = self.content_color * self.content_alpha
    #         self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
    #     elif imgdict.get('content-opaque','none') != 'none':
    #         content = load_img_torch(path + imgdict['content-opaque'], device, "alpha")
    #         self.content_color = content[:,:3]
    #         self.content_alpha = content[:,3:]
    #         self.ovl_black = self.content_color * self.content_alpha
    #         self.ovl = self.content_color * self.content_alpha + self.bg * (1 - self.content_alpha)
    #         self.content_color = self.ovl
    #         self.content_alpha = torch.ones_like(self.content_alpha, device=device, dtype=torch.float32)
    #     else:
    #         sys.exit("Error: input must have content or opaque image path")
        
    #     if imgdict.get('blend','none') != 'none':
    #         self.blend = load_img_torch(path + imgdict['blend'], device)
        
    #     if imgdict.get('mask','none') == 'none':
    #         if self.content_alpha == None:
    #            self. mask = torch.ones_like(self.bg[:,:1,:,:], device=device, dtype = torch.float32)
    #         else:
    #             self.mask = torch.where(self.content_alpha > 0,1.,0.).to(torch.float32)
    #     elif imgdict.get('mask','none') == 'full':
    #         self.mask = torch.ones_like(self.bg[:,:1,:,:], device=device, dtype = torch.float32)
    #     else:
    #         self.mask = load_img_torch(path + imgdict['mask'], device, "gray")
        
    #     if imgdict.get('vismap','none') != 'none':
    #         self.vismap = load_img_torch(path + imgdict['vismap'], device, "gray")
    #     else:
    #         if imgdict.get('vis_value','none') != 'none':
    #             assert 0. <= imgdict.get('vis_value','none') <= 1.0
    #             self.vismap = torch.ones_like(self.bg[:,:1,:,:], device=device, dtype = torch.float32) * imgdict.get('vis_value','none')

def load_stimulus(img_list: list[dict], path: str, device: str) -> list[stimulus]:
    stimulus_list = []
    for ind, imgdict in enumerate(img_list): # output is bgr
        stim = stimulus()
        stim.set_from_imgdict(imgdict, path, device)
        stim.set_index([str(ind)])
        stimulus_list.append(stim)
    return stimulus_list

class stimulusVideo():
    def __init__(self):
        self.index: int | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.bg_video: cv2.VideoCapture | None = None
        self.ovl_video: cv2.VideoCapture | torch.Tensor | None = None
        self.mask_video: cv2.VideoCapture | None = None
        self.vismap_video: cv2.VideoCapture | torch.Tensor | None = None
        self.obj_vis_value: float | None = None
        self.else_vis_value: float | None = None
        self.bg_flip: bool = False
        self.num_frame: int | None = None
        self.fps: int | None = None
    
    def set_index(self, index: int):
        self.index = index
    
    def get_original_size(self) -> (int, int):
        if type(self.ovl_video) == cv2.VideoCapture:
            return (self.ovl_video.get(cv2.CAP_PROP_FRAME_WIDTH), self.ovl_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            return (self.ovl_video.shape[2], self.ovl_video.shape[3])
    
    def set_from_imgdict(self, imgdict: dict, path: str, device: str):
        print(imgdict)
        self.bg_video = cv2.VideoCapture(path + imgdict['bg'])
        num_frame_bg = self.bg_video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.bg_video.get(cv2.CAP_PROP_FPS)
        print(f"num_frames_bg: {num_frame_bg}")

        if imgdict['overlaid'].endswith('.mp4') or imgdict['overlaid'].endswith('.mov') or imgdict['overlaid'].endswith('.m4v'):
            self.ovl_video = cv2.VideoCapture(path + imgdict['overlaid'])
            num_frame_ovl = self.ovl_video.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            self.ovl_video = load_img_torch(path + imgdict['overlaid'], device, "alpha")
            num_frame_ovl = None
        
        if imgdict.get('mask','none') != 'none':
            
            if imgdict['mask'].endswith('.mp4') or imgdict['mask'].endswith('.mov') or imgdict['mask'].endswith('.m4v'):
                self.mask_video = cv2.VideoCapture(path + imgdict['mask'])
            else:
                self.mask_video = load_img_torch(path + imgdict['mask'], device, "gray")
        
        if imgdict.get('target_vismap','none') == 'none':
            assert imgdict.get('obj_vis_value', None) != None
        else:
            if imgdict['target_vismap'].endswith('.mp4') or imgdict['target_vismap'].endswith('.mov') or imgdict['target_vismap'].endswith('.m4v'):
                self.vismap_video = cv2.VideoCapture(path + imgdict['target_vismap'])
            else:
                self.vismap_video = load_img_torch(path + imgdict['target_vismap'], device, "gray")
        self.obj_vis_value = imgdict.get('obj_vis_value', None)
        self.else_vis_value = imgdict.get('else_vis_value', None)

        self.num_frame = imgdict.get('num_frame', None)
        if self.num_frame == None:
            seconds = imgdict.get('seconds', None)
            if seconds != None:
                self.num_frame = min(self.fps * seconds, num_frame_bg)
            else:
                if num_frame_ovl == None or num_frame_ovl > num_frame_bg:
                    self.num_frame = int(num_frame_bg)
                else:
                    self.num_frame = int(num_frame_ovl)
            print("num_frame_ovl: ", num_frame_ovl)
            print("num_frame_bg: ", num_frame_bg)
            print("num frame is set as", self.num_frame)
        
        (org_width, org_height) = self.get_original_size()
        self.width = imgdict.get('width', org_width)
        self.height =  imgdict.get('height', org_height)
        self.bg_flip = imgdict.get('tg_flip', False)
        self.shrink = imgdict.get('shrink', None)

def load_video_stimulus(img_list: list[dict], path: str, device: str) -> list[stimulusVideo]:
    stimulus_list = []
    for ind, imgdict in enumerate(img_list):
        stim = stimulusVideo()
        stim.set_from_imgdict(imgdict, path, device)
        stim.set_index(ind)
        stimulus_list.append(stim)
    return stimulus_list

def nearest_resize(data:torch.Tensor, height: int, width:int) -> torch.Tensor:
    resizer = T.Resize(size=(height, width), interpolation=T.InterpolationMode.NEAREST)
    return resizer(data)

def bicubic_resize(data:torch.Tensor, height: int, width:int) -> torch.Tensor:
    resizer = T.Resize(size=(height, width), interpolation=T.InterpolationMode.BICUBIC)
    return resizer(data)

def bilinear_resize(data:torch.Tensor, height: int, width:int) -> torch.Tensor:
    resizer = T.Resize(size=(height, width), interpolation=T.InterpolationMode.BILINEAR)
    return resizer(data)

def shrink_and_center(image: torch.Tensor, shrink: float = None, boundary="zero") -> torch.Tensor:
    if shrink is None:
        return image  # shrink が None の場合は処理をスキップ
    
    _, c, h, w = image.shape  # 画像のチャンネル数、高さ、幅を取得
    new_h, new_w = int(h * shrink), int(w * shrink)  # 縮小後のサイズを計算
    
    # 画像をリサイズ
    resized_image = bilinear_resize(image, new_h, new_w)
    
    # 貼り付ける開始位置を計算
    start_h, start_w = (h - new_h) // 2, (w - new_w) // 2

    if boundary=="replicate":
        # 元のサイズの画像を作成（端の画素値を繰り返す）
        output = F.pad(resized_image, (start_w, w - (start_w + new_w), start_h, h - (start_h + new_h)), mode='replicate')
    else:
        # ゼロで初期化した元のサイズの画像を作成
        output = torch.zeros_like(image)
        # 中心に配置
        output[:, :, start_h:start_h + new_h, start_w:start_w + new_w] = resized_image
        
    return output

def save_img_torch(path: str, data: torch.Tensor, idx: int | None = 0):
    assert len(data.shape) in [3, 4]
    MIN_HEIGHT = 256
    if len(data.shape) == 3:
        height = data.shape[1]
        width = data.shape[2]
    else:
        height = data.shape[2]
        width = data.shape[3]
    if height < MIN_HEIGHT:
        data = nearest_resize(data, MIN_HEIGHT, int(MIN_HEIGHT * (width/height)))
    if idx != None:
        data = data[idx]
    data = data.detach().cpu().numpy()
    data = np.transpose(data, [1,2,0])
    cv2.imwrite(path,np.uint8(255*data))

def load_img_torch(path: str, device: str, flag: str = "color") -> torch.Tensor:
    assert flag in ["color", "alpha", "gray"]
    if flag == "alpha":
        data = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    elif flag == "gray":
        data = cv2.imread(path,0)
        data = np.expand_dims(data, -1)
    else:
        data = cv2.imread(path)
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    data = data / 255.
    data = torch.permute(data, (2, 0, 1)).unsqueeze(0)
    return data

def load_img_list(base_path: str, names: list[str], device:torch.device, ext:str = "png", flag: str = "color") -> list[dict]:
    assert flag in ["color", "alpha"]
    dict_list = []
    for ind, name in enumerate(names):
        filename = base_path + name + "." + ext
        content: torch.Tensor = load_img_torch(filename, device, flag)
        alpha: torch.Tensor | None = None
        if flag == "alpha":
            alpha = content[:,3:]
            content = content[:,:3]
        else:
            shape = content.shape
            alpha = torch.ones((shape[0],1,shape[2],shape[3]), dtype=torch.float32, device=device)
        dict_list.append({"name": name.replace('/', ''), "img": content, "alpha": alpha})
    return dict_list

def inv_sigmoid(Y: torch.Tensor, param: np) -> torch.Tensor:
    A = torch.as_tensor(param[0],dtype=torch.float64)
    B = torch.as_tensor(param[1],dtype=torch.float64)
    v = torch.as_tensor(param[2],dtype=torch.float64)
    Y = torch.as_tensor(torch.clamp(Y*4+1,min=1,max=4.99999),dtype=torch.float64)
    Q = ((5-A)/(1-A))**v - 1
    X = -torch.log((((5-A)/(Y-A))**v - 1)/Q)/B

    return torch.as_tensor(X,dtype=torch.float32)

def generalized_sigmoid(X: torch.Tensor, param: np) -> torch.Tensor:
    A = param[0]
    B = param[1]
    v = param[2]
    Q = ((5-A)/(1-A))**v - 1
    Y = A + (5-A)/((1+Q*torch.exp(-B*X))**(1/v))
    return (Y-1)/4

def save_grayimg_plt(path: str,
                     data: torch.Tensor,
                     idx: int = 0,
                     norm: bool = False,
                     title: str = None,
                     cmap: str = None,
                     center: str = "none"):
    center in ["none", "twoslope", "minmax"]
    write_image = data[idx,0].detach().cpu().numpy()
    max_data = np.abs(write_image).max()
    fig, ax = plt.subplots()
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    if norm:
        im = ax.imshow(write_image, vmin=0, vmax=1, cmap = cmap)
    else:
        if center == "twoslope":
            im = ax.imshow(write_image, cmap = cmap, norm = colors.TwoSlopeNorm(0))
        elif center == "minmax":
            im = ax.imshow(write_image, cmap = cmap, vmin=max_data, vmax=-1*max_data)
        else:
            im = ax.imshow(write_image, cmap = cmap)
    fig.colorbar(im, cax=cax)
    if title:
        ax.set_title(title, fontsize=20)
    ax.tick_params(labelbottom = False, labelleft = False, bottom=False, left=False)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(path, transparent=True)
    plt.clf()
    plt.close()

def printImgCore(data: np.ndarray,
                 filename: str,
                 cmap: str = 'gray',
                 scale: tuple[float] | None = None,
                 onlyImg = False):
    if onlyImg:
        fig, ax = plt.subplots(figsize=(data.shape[1], data.shape[0]), dpi=1)
        if scale != None:
            vmin,vmax = scale 
            plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(data, cmap=cmap)
        plt.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        fig, ax = plt.subplots()
        if scale != None:
            vmin,vmax = scale 
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            im = ax.imshow(data, cmap=cmap)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()

def make_checkboard(width: int, height: int, device: torch.device, check_size:int = 15, color:int = 0.5) -> torch.Tensor:
    check_board = torch.ones((width, height), dtype = torch.float32, device = device)
    i=0
    while i * check_size < height:
        top = i * check_size
        bottom = min(height, (i+1)*check_size)
        j=0
        while j * check_size < width:
            left = j * check_size
            right = min(width, (j+1)*check_size)
            if (i+j)%2 == 1 :check_board[left:right,top:bottom] = color
            j += 1
        i += 1

    return check_board
