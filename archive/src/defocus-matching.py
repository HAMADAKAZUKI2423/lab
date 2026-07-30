## This script calculates the optical factor of human eye based on photometry 
import torch
import pycvvdp
import pycvvdp.display_model as display_model
from pycvvdp.video_source import reshuffle_dims
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import math
deg2rad = math.pi/180
rad2deg = 180/math.pi
mm, um, nm= 1e-3, 1e-6, 1e-9

def pupil_d_unified(L,area,age):
    ## This function calculates the pupil diameter in mm with a respect to 
    ## L: luminance (cd/m^2)
    ## area: area
    ## age: user age
    clamp = lambda x, max_val, min_val: max(min(x,max_val),min_val)

    y0 = 28.58 # reference age from the reference
    y = clamp(age, 20, 83)
   
    La = L * area 
    pd_sd =   7.75 - 5.75 * ((La/846)**(0.41) / ((La/846)**0.41+2))
    pd = pd_sd + (y-y0)*(0.02132 - 0.009562*pd_sd)

    return pd 

## Fourier transform of torch tensor
def FT2(tensor):
    """ Perform 2D fft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
    tensor_ft = torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))
    return tensor_ft


def iFT2(tensor):
    """ Perform 2D ifft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
    tensor_ift = torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))
    return tensor_ift

def gaussian(x, mu, sigma):
    """
    Compute the Gaussian function.
    """
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def preview_enlarged_image(f, crop_window=None, center=None, scale=None, num_tick=None, xlabel=None, ylabel=None, title=None):
    '''
    this function previews the cropped image of input tensor with size of (BCFHW: 1C1HW)
    '''
    
    f_np = np.rollaxis(f.cpu().squeeze().numpy(),0,3) 
    if center is None:
        center = (f.shape[-2]//2, f.shape[-1]//2)
    if crop_window is None:
        crop_window = (f.shape[-2],f.shape[-1])
    f_crop = f_np[center[0]-crop_window[0]//2:center[0]+crop_window[0]//2+1, \
                            center[1]-crop_window[1]//2:center[1]+crop_window[1]//2+1,:]
    plt.imshow(f_crop/np.max(f_crop))
    if num_tick is not None and scale is not None:
        ax = plt.gca()
        xx = scale[1]*np.arange(-(crop_window[1]//2), crop_window[1]//2 + 1, crop_window[1]//(num_tick-1))
        yy = scale[0]*np.arange(-(crop_window[0]//2), crop_window[0]//2 + 1, crop_window[0]//(num_tick-1))
        ax.set_xticks(np.arange(0, crop_window[1] + 1, crop_window[1]//(num_tick-1)))
        ax.set_yticks(np.arange(0, crop_window[0] + 1, crop_window[0]//(num_tick-1)))
        ax.set_xticklabels(np.round(xx))
        ax.set_yticklabels(np.round(yy))
    
    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.show()

class optics_model:
    def __init__(self, age= None, pd=None, test_frame=None, bkg_frame = None, test_dp=None, test_dg=None, bkg_dp=None, bkg_dg=None, device=None,Num_layer=2, blur_scale_factor=0.55):
        
        test_frame = test_frame.squeeze() ## C x H x W
        bkg_frame = bkg_frame.squeeze() ## C x H x W
        
        self.test_frame = test_frame
        self.bkg_frame = bkg_frame
        self.test_dp = test_dp
        self.test_dg = test_dg
        self.bkg_dp = bkg_dp
        self.bkg_dg = bkg_dg
        self.device = device
        self.blur_scale_factor = blur_scale_factor
        
        if isinstance(Num_layer, int):
            self.Num_layer = Num_layer
        else:
            RuntimeError('Define integer for number of layers') 

        self.resolution = (test_frame.shape[-2], test_frame.shape[-1]) ## Resolution of test frame (H,W)
        self.size_m = (self.test_dg.display_size_m[-1]/self.test_dg.resolution[-1]*self.resolution[-2], self.test_dg.display_size_m[0]/self.test_dg.resolution[0]*self.resolution[-1]) # (H,W)
        self.display_size_deg = (2*math.atan(self.size_m[0]/self.test_dg.distance_m/2)*rad2deg, 2*math.atan(self.size_m[1]/self.test_dg.distance_m/2)*rad2deg) # (H,W)
        self.area = self.display_size_deg[0] * self.display_size_deg[1] ## Area in deg^2

        self.set_luminance() ## Set luminance based on the display model
        self.set_pd(pd,age) ## Set pupil diameter based on luminance and age

        self.set_dioptric_distance()
        self.set_angle_domain()

    def set_dioptric_distance(self):
        self.D =1/self.test_dg.distance_m - 1/self.bkg_dg.distance_m

    def set_luminance(self):
        '''
        Calculate the luminance based on the display model
        '''
        test_rgb2y = self.test_dp.rgb2xyz_list[1]
        bkg_rgb2y = self.bkg_dp.rgb2xyz_list[1]

        tl = torch.mean(self.test_frame[0,:,:]*test_rgb2y[0] + self.test_frame[1,:,:]* test_rgb2y[1]+self.test_frame[2,:,:]*test_rgb2y[2])
        if len(self.bkg_frame.shape) ==2:
            bl = torch.mean(self.bkg_frame)
        else:
            bl = torch.mean(self.bkg_frame[0,:,:]*bkg_rgb2y[0] + self.bkg_frame[1,:,:]*bkg_rgb2y[1]+self.bkg_frame[2,:,:]*bkg_rgb2y[2])

        self.luminance = (tl+bl).float()

    def set_pd(self,pd=None,age=None):
        '''
        Set pupil diameter based on mean luminance and age
        '''
        if pd is None and age is not None:
            self.pd = pupil_d_unified(self.luminance, self.area, age)
        elif age is None:
            self.pd = pupil_d_unified(self.luminance, self.area, 30)
        if pd is not None:
            self.pd = pd 

    def set_angle_domain(self):
        w = self.test_frame.shape[-1]
        h = self.test_frame.shape[-2] 
        x_deg = torch.linspace(-self.display_size_deg[1]/2,self.display_size_deg[1]/2,w+1).to(self.device) # To include 0 
        y_deg = torch.linspace(-self.display_size_deg[0]/2,self.display_size_deg[0]/2,h+1).to(self.device)
        self.Y_deg, self.X_deg = torch.meshgrid(y_deg, x_deg)

    def calculate_psf_ray(self, D= None,wavelengths=None, is_preview=False):
        '''
        Calculate the psf based on geometric optics (using circle of confusion)
        wavelengths: tuple of wavelength (r,g,b) in a unit of nm
        '''
        w = self.test_frame.shape[-1]
        h = self.test_frame.shape[-2] 
        Y_deg = self.Y_deg
        X_deg = self.X_deg
        psf = torch.zeros_like(self.test_frame)
        if D is None:
            D = self.D

        if wavelengths is None:
            bd_deg = rad2deg*D*self.pd*mm
            if ( bd_deg.device!='cpu'):
                bd_deg = bd_deg.to('cpu')
            sigma = self.blur_scale_factor * bd_deg / 2 ## Matching size based on Chromablur paper, scaled by blur_scale_factor
            tmp_psf = gaussian(torch.sqrt(X_deg**2 + Y_deg**2), 0, sigma)
            tmp_psf = F.interpolate(tmp_psf.unsqueeze(0).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False).squeeze()
            tmp_psf = tmp_psf / torch.sum(tmp_psf) 
            tmp_psf = reshuffle_dims(tmp_psf, in_dims = "HW", out_dims = "BCFHW") ## Making the domain shape identical
            psf = tmp_psf.repeat(1,3,1,1,1) ## expand in color channel
        else: # Implement chromatic defocus
            raise RuntimeError('Define proper wavelengths')   
        
        psf=psf.to(self.device)
        if is_preview:
            preview_enlarged_image(psf, crop_window=(21,21), \
                        scale=(60*self.display_size_deg[0]/h, 60*self.display_size_deg[1]/w), \
                        num_tick=3, \
                        xlabel ='angle[arcmin]',\
                            ylabel='angle[armin]',\
                                title = f'pd: {self.pd:.2f} mm / D: {D:.2f} D')
        return psf

    def gen_psf(self, PREVIEW = False):
        psf = torch.zeros(size=(self.Num_layer,3,1,*self.resolution)).to(self.device)
        D_vec = torch.linspace(0,self.D, self.Num_layer)

        for d in range(self.Num_layer):
            psf[d,:,:,:,:] =  self.calculate_psf_ray(D = D_vec[d], is_preview=PREVIEW) 

        return psf


    def get_blur_image(self,bkg_frame, psf, is_preview =False):            
        blur_frame = torch.zeros(size=(1,3,1,*self.resolution),device=self.device) 

        if len(psf.size())==4: ## In case of dimension reduction 
            psf = psf.unsqueeze(0)

        for c in range(3):
            if (bkg_frame.shape[1]==1): # gray scale input
                blur_frame[:,c,:,:,:] = torch.abs(iFT2(FT2(bkg_frame)*FT2(psf[:,c,:,:,:])))
                blur_frame[:,c,:,:,:] = blur_frame[:,c,:,:,:] *torch.sum(bkg_frame)/ torch.sum(blur_frame[:,c,:,:,:]) # Energy preservation
            else:
                blur_frame[:,c,:,:,:] = torch.abs(iFT2(FT2(bkg_frame[:,c,:,:,:])*FT2(psf[:,c,:,:,:])))
                blur_frame[:,c,:,:,:] = blur_frame[:,c,:,:,:] *torch.sum(bkg_frame[:,c,:,:,:] )/ torch.sum(blur_frame[:,c,:,:,:]) # Energy preservation
        if is_preview:
            preview_enlarged_image(blur_frame, crop_window=(101,101), center=(270,580),\
                                   title= f'pd: {self.pd:.2f} mm / D: {self.del_D:.2f} D')

        return blur_frame
        
    def get_image(self, frame, bkg_frame, psf, fs_weight=None, mode = 'sum', idx =None):
        image = torch.zeros_like(frame)
        if len(psf.size())==4: ## In case of dimension reduction 
            psf = psf.unsqueeze(0)

        if fs_weight is None: # Put weight on the foreground
            fs_weight = torch.zeros(size=(self.Num_layer,))

        if len(fs_weight) != self.Num_layer:
           raise RuntimeError('Wrong inputs for the focal stack weight')

        if mode == 'sum':
            loop_range = self.Num_layer
        elif mode == 'individual':
            loop_range = 1
        else:
            raise RuntimeError('Please pick the recon mode')
            
        for d in range(loop_range):
            if mode=='individual':
                d = idx
                fs_weight[d] = 1

            if d==0:
                image = image + fs_weight[d] *(frame + self.get_blur_image(bkg_frame,psf[self.Num_layer-1-d,:,:,:,:]))
            elif d==self.Num_layer-1:
                image = image + fs_weight[d] *(self.get_blur_image(frame,psf[d,:,:,:,:]) + bkg_frame)
            else:
                image = image + fs_weight[d] *(self.get_blur_image(frame,psf[d,:,:,:,:]) + self.get_blur_image(bkg_frame,psf[self.Num_layer-1-d,:,:,:,:]))

        return image

if __name__ == '__main__':
    # ================================================================
    # スライダーでぼけ具合（ディオプトリ差 D）をインタラクティブに変更するデモ
    # から、画像のぼけ具合を調整し、中央の枠と比較するデモへ変更
    # ================================================================
    import matplotlib.image as mpimg
    from PIL import Image

    # --- 1. デモ用の設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 画像と表示の設定 ---
    # TODO: 表示したい画像のパスに変更してください
    IMG_PATH = 'path/to/your/image.png' # 表示する画像のパス
    VIEWING_DISTANCE_CM = 57  # 観察距離 (cm)
    FRAME_VISUAL_ANGLE_DEG = 5.0 # 中央の枠の視角 (degree)
    IMAGE_VISUAL_ANGLE_DEG = 5.0 # 画像の視角 (degree)
    CANVAS_SIZE_PX = 512 # 全体の画像サイズ (pixel)

    # ダミーのディスプレイ設定
    class DummyDisplay:
        def __init__(self, distance_cm, resolution_wh):
            self.distance_m = distance_cm / 100.0
            self.resolution = resolution_wh # (W, H)
            pixel_pitch_m = 0.00023 # 1ピクセルが0.23mmと仮定 (一般的なモニタ)
            self.display_size_m = (resolution_wh[0] * pixel_pitch_m, resolution_wh[1] * pixel_pitch_m) # (width_m, height_m)
            self.rgb2xyz_list = [0, [0.2126, 0.7152, 0.0722]] # sRGBの輝度係数

    # --- 2. optics_model の初期化 ---
    # この時点でのフレームはダミーでOK
    dummy_frame = torch.zeros(1, 3, 1, CANVAS_SIZE_PX, CANVAS_SIZE_PX, device=device)
    model = optics_model(
        age=30,
        test_frame=dummy_frame,
        bkg_frame=dummy_frame,
        test_dg=DummyDisplay(distance_cm=VIEWING_DISTANCE_CM, resolution_wh=(CANVAS_SIZE_PX, CANVAS_SIZE_PX)),
        bkg_dg=DummyDisplay(distance_cm=VIEWING_DISTANCE_CM * 2, resolution_wh=(CANVAS_SIZE_PX, CANVAS_SIZE_PX)), # bkgは遠方と仮定
        device=device
    )
    model.pd = 4.0 # 瞳孔径を4mmに固定

    # --- 3. 表示要素の準備 ---
    
    # 視角からピクセルサイズを計算
    display_height_deg = model.display_size_deg[0]
    pixels_per_deg = CANVAS_SIZE_PX / display_height_deg
    frame_size_px = int(FRAME_VISUAL_ANGLE_DEG * pixels_per_deg)
    image_size_px = int(IMAGE_VISUAL_ANGLE_DEG * pixels_per_deg)

    # (1) ぼかす対象の画像を読み込んで配置
    try:
        img_np = mpimg.imread(IMG_PATH)
        # チャンネル数とデータ型を正規化
        if len(img_np.shape) == 2: # Grayscale
            img_np = np.stack([img_np]*3, axis=-1)
        if img_np.shape[2] == 4: # RGBA
            img_np = img_np[:, :, :3]
        if img_np.dtype == np.uint8: # 0-255 to 0-1
            img_np = img_np / 255.0
    except (FileNotFoundError, OSError):
        print(f"警告: 画像ファイルが見つかりません '{IMG_PATH}'. ダミーの市松模様を使用します。")
        # ダミー画像（チェッカーボード）を作成
        c = np.zeros((image_size_px, image_size_px))
        s = max(1, image_size_px // 8)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    c[i*s:(i+1)*s, j*s:(j+1)*s] = 1
        img_np = np.stack([c, c, c], axis=-1)

    # PILを使ってリサイズ
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil = img_pil.resize((image_size_px, image_size_px), Image.LANCZOS)
    img_resized_np = np.array(img_pil) / 255.0

    # キャンバス上に画像を配置
    image_plane_np = np.zeros((CANVAS_SIZE_PX, CANVAS_SIZE_PX, 3))
    center_y, center_x = CANVAS_SIZE_PX // 2, CANVAS_SIZE_PX // 2

    # ご要望に基づき、画像を白い枠と重なるように中央に配置します
    img_top_y = center_y - image_size_px // 2
    img_left_x = center_x - image_size_px // 2
    image_plane_np[img_top_y:img_top_y + image_size_px, img_left_x:img_left_x + image_size_px, :] = img_resized_np

    image_plane_torch = torch.from_numpy(image_plane_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().to(device)

    # (2) 中央の四角い枠を作成
    frame_plane_np = np.zeros((CANVAS_SIZE_PX, CANVAS_SIZE_PX, 3))
    y0, x0 = center_y - frame_size_px // 2, center_x - frame_size_px // 2
    y1, x1 = center_y + frame_size_px // 2, center_x + frame_size_px // 2
    line_width = 2 # 枠の線の太さ
    frame_plane_np[y0:y1, x0:x0+line_width, :] = 1.0 # Left
    frame_plane_np[y0:y1, x1-line_width:x1, :] = 1.0 # Right
    frame_plane_np[y0:y0+line_width, x0:x1, :] = 1.0 # Top
    frame_plane_np[y1-line_width:y1, x0:x1, :] = 1.0 # Bottom
    frame_plane_torch = torch.from_numpy(frame_plane_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().to(device)


    # --- 4. Matplotlib UI のセットアップ ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # 前景画像の表示状態を管理するフラグ
    image_visible = True

    # 初期状態の画像を表示
    initial_D = 0.0 # 初期ディオプトリ
    psf = model.calculate_psf_ray(D=initial_D)
    blurred_img_torch = model.get_blur_image(image_plane_torch, psf)
    
    # 枠と合成
    combined_img_torch = blurred_img_torch + frame_plane_torch
    
    # 表示用に画像をnumpy配列に変換
    img_to_show = combined_img_torch.cpu().squeeze().permute(1, 2, 0).numpy()
    img_to_show = np.clip(img_to_show, 0, 1) # 念のためクリッピング
    
    im = ax.imshow(img_to_show)
    ax.set_title(f'Defocus Matching (D = {initial_D:.2f})')
    ax.axis('off') # 軸を非表示に

    # スライダーを追加
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    d_slider = Slider(ax=ax_slider, label='Diopter (D)', valmin=0.0, valmax=2.0, valinit=initial_D)
    plt.text(0.5, -0.8, '← : Less Blur / → : More Blur,   ↑ / ↓: Toggle Image',
             horizontalalignment='center', verticalalignment='center', transform=ax_slider.transAxes)

    # スライダーが動いたときの処理
    def update(val):
        D = d_slider.val
        psf = model.calculate_psf_ray(D=D)
        blurred_img_torch = model.get_blur_image(image_plane_torch, psf)

        # 枠と合成
        if image_visible:
            combined_img_torch = blurred_img_torch + frame_plane_torch
        else:
            combined_img_torch = frame_plane_torch.clone()

        img_to_show = combined_img_torch.cpu().squeeze().permute(1, 2, 0).numpy()
        img_to_show = np.clip(img_to_show, 0, 1)
        
        im.set_data(img_to_show)
        ax.set_title(f'Defocus Matching (D = {D:.2f})')
        fig.canvas.draw_idle()

    d_slider.on_changed(update)

    def on_key_press(event):
        global image_visible
        step = 0.05  # スライダーの1ステップの変化量

        if event.key == 'up' or event.key == 'down':
            image_visible = not image_visible
            update(d_slider.val)
        elif event.key == 'right':  # 値を増やす
            new_val = min(d_slider.valmax, d_slider.val + step)
            d_slider.set_val(new_val)
        elif event.key == 'left':  # 値を減らす
            new_val = max(d_slider.valmin, d_slider.val - step)
            d_slider.set_val(new_val)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()