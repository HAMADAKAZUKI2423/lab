import torch
from torch import nn
from .encoders import InvertedResidualEncoder
from torchvision.models.segmentation.deeplabv3 import ASPP
from .decoders import SimpleDecoder#, FilmDecoder2
# from .film import FilmMlp2
from .INetwork import INetwork
from .reflectionnet import InvertedResidualEncoderReflection, ASPPReflection, SimpleDecoderReflection
import torch.nn.functional as F
import torchvision.transforms as T

class TestNet(INetwork):
    
    def __init__(self, out_channels = 1, mode = 0, blur_mode=False, blur_maxlevel=6, base_alpha=0.5):
        super(TestNet, self).__init__()
        self.mode = mode
        self.blur_mode = blur_mode
        
        if mode == 0:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 64, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(7, inverted_residual_setting)
            self.aspp = ASPP(inverted_residual_setting[-1][1], [3, 6, 9], 128)
            self.decoder = SimpleDecoder([128, 32, 24, 16, out_channels], [64, 32, 24, 16])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        elif mode == 1:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 64, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(7, inverted_residual_setting)
            self.decoder = SimpleDecoder([0, 32, 24, 16, out_channels], [64, 32, 24, 16])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        elif mode == 2:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 8, 1, 1],
                [6, 12, 1, 2],
                [6, 16, 1, 2],
                [6, 32, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(7, inverted_residual_setting)
            self.aspp = ASPP(inverted_residual_setting[-1][1], [3, 6, 9], 64)
            self.decoder = SimpleDecoder([64, 16, 12, 8, out_channels], [32, 16, 12, 8])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        elif mode == 3:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 8, 1, 1],
                [6, 12, 1, 2],
                [6, 16, 1, 2],
                [6, 32, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(7, inverted_residual_setting)
            self.decoder = SimpleDecoder([0, 16, 12, 8, out_channels], [32, 16, 12, 8])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        elif mode == 4:
            # reflection padding, localized average pooling, reduced dilation rate (receptive field: about 96)
            
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 64, 1, 2],
            ]

            self.backbone = InvertedResidualEncoderReflection(7, inverted_residual_setting)
            # self.aspp = ASPPReflection(inverted_residual_setting[-1][1], [3, 6, 9], 128)
            self.aspp = ASPPReflection(inverted_residual_setting[-1][1], [2, 4, 6], 128)
            self.decoder = SimpleDecoderReflection([128, 32, 24, 16, out_channels], [64, 32, 24, 16])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        elif mode == 5:
            # ASPPのwindowだけ小さくした
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 64, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(7, inverted_residual_setting)
            self.aspp = ASPP(inverted_residual_setting[-1][1], [2, 4, 6], 128)
            self.decoder = SimpleDecoder([128, 32, 24, 16, out_channels], [64, 32, 24, 16])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
        
        if self.blur_mode:
            filt = self.gauss_kernel(3)
            self.register_buffer("filt", filt)
            self.pad_two = nn.ReflectionPad2d(2)
            self.blur_maxlevel = blur_maxlevel
            self.base_alpha = base_alpha
    
    def forward(self, x1, x2, tv, resize_scale = 1.0, get_levelmap=False):

        if self.blur_mode:
            data_height = x1.shape[2]
            data_width = x1.shape[3]
            calc_height = int(data_height * resize_scale)
            calc_width = int(data_width * resize_scale)

            calc_x1 = T.functional.resize(x1, size=(calc_height, calc_width), antialias = True)
            calc_x2 = T.functional.resize(x2, size=(calc_height, calc_width), antialias = True)
            if len(tv.shape) > 3:
                calc_tv = T.functional.resize(tv, size=(calc_height, calc_width), antialias = True)
            else:
                calc_tv = tv
        
        else:
            calc_x1 = x1
            calc_x2 = x2
            calc_tv = tv

        calc_tv = calc_tv * 2.0-1.0
        x = torch.cat([calc_x1,calc_x2,calc_tv],dim=1)
        x = self.backbone(x)
        if self.mode in [0,2,5]:
            x_aspp = self.aspp(x[-1])
            x = self.decoder([*x, x_aspp])
        elif self.mode in [4]:
            x_aspp = self.aspp(x[-1])
            x = self.decoder(x_aspp, x[-1], x[-2], x[-3], x[-4])
        else:
            x = self.decoder(x)
        x = self.binding(x)

        if self.blur_mode:
            x = T.functional.resize(img=x, size=(data_height, data_width), antialias = True)

            if get_levelmap:
                return x
            else:
                # filter reference image (x2)
                ref_Gpyr = self.get_Upsampled_Gpyr(x2, self.blur_maxlevel)
                filtered_ref = self.blend_with_soft_levels(ref_Gpyr, x.squeeze(1))
                return filtered_ref

        else:
            return x
    
    # 以下はadaptive blur blend用
    def gauss_kernel(self,channels: int=3):
        kernel = torch.tensor([
                [1, 5, 8, 5, 1],
                [5, 25, 40, 25, 5],
                [8, 40, 64, 40, 8],
                [5, 25, 40, 25, 5],
                [1, 5, 8, 5, 1]],dtype=torch.float32)
        kernel /= 400.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    
    def get_Upsampled_Gpyr(self, image: torch.Tensor, level: int):
        #Gaussian pyramid upsampled by one level
        
        J = image
        dims = image.shape[1]
        #pyr = []
        gpyr=[J]
        for i in range(level):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                            groups=dims)
            #I_up = self.upsample(I)#include conv
            gpyr.append(I)

            J = I
        
        up_pyr = []
        for i, g_img in enumerate(gpyr):
            for j in range(i):
                g_img = self.upsample(g_img)
            up_pyr.append(g_img)
        up_pyr = torch.stack(up_pyr,dim=1)

        return up_pyr
    
    def soft_level_weights(self,
        level_map: torch.Tensor,  # (N,H,W) 実数レベル (たとえば [0, L-1] にクランプなど)
        L: int
    ) -> torch.Tensor:
        """
        三角形補間に基づく「レベル毎のソフトな重み」をバッチ対応で計算する。

        入力:
        level_float_map: (N,H,W) - ネットワーク等から出力された「連続レベル」マップ
        L: レベル数 (0,1,...,L-1)

        出力:
        weights: (N, L, H, W)
            各ピクセルがレベル i に割り当てる重み w_i。
            三角形補間により、floor(l) と ceil(l) 周辺だけが非ゼロ。
            各ピクセルについて ∑_i w_i = 1.
        """

        level_float_map = level_map * (L-1)

        device = level_float_map.device
        dtype = level_float_map.dtype

        N, H, W = level_float_map.shape

        # i = 0..L-1 を (L,) で用意 → (1,L,1,1) にreshape
        i_vals = torch.arange(L, device=device, dtype=dtype).view(1, L, 1, 1)  # (1,L,1,1)

        # level_float_map: (N,H,W) → (N,1,H,W)
        l_map_4d = level_float_map.unsqueeze(1)  # (N,1,H,W)

        # 三角形関数: tmp = 1 - |l_map - i|
        # 形状: (N,L,H,W)
        tmp = 1.0 - (l_map_4d - i_vals).abs()

        # 負値は0にする(ReLU)
        tmp = F.relu(tmp)  # max(0, x)

        # 正規化 (レベル軸 L 方向の合計が1になるように)
        # sum_tmp: (N,H,W)
        sum_tmp = tmp.sum(dim=1, keepdim=False) + 1e-8

        # (N,L,H,W)
        weights = tmp / sum_tmp.unsqueeze(1)
        return weights


    def blend_with_soft_levels(self,
        pyramid: torch.Tensor,  # 各レベル画像を格納したTensor (N,L,C,H,W)
        level_float_map: torch.Tensor # (N,H,W) 連続値 (たとえば [0, L-1])
    ) -> torch.Tensor:
        """
        ピラミッド各レベル (N,C,H,W) を、三角形補間に基づくソフトウェイトでブレンドする。

        入力:
        pyramid[i]: (N,C,H,W)  - レベル i の画像 (全て同じ (H,W) にアップサンプル済み想定)
        level_float_map: (N,H,W) - ピクセルごとの連続レベル指標

        出力:
        out: (N,C,H,W) - ブレンド画像
        """
        # レベル数
        L = pyramid.shape[1]

        # (N,L,H,W) の重みマップを計算
        w = self.soft_level_weights(level_float_map, L)  # (N,L,H,W)

        # pyramid は各要素が (N,C,H,W) → stack でまとめる: (L,N,C,H,W)
        # stacked = torch.stack(pyramid, dim=0)  # (L,N,C,H,W)

        # # (L,N,C,H,W) → (N,L,C,H,W) に permute
        # stacked = stacked.permute(1, 0, 2, 3, 4)  # (N,L,C,H,W)

        # 重み w: (N,L,H,W) → (N,L,1,H,W) に reshape
        w_5d = w.unsqueeze(2)  # ブロードキャストのため (N,L,1,H,W)

        # 要素積してレベル軸 L を足し合わせる: (N,C,H,W)
        out = (pyramid * w_5d).sum(dim=1)  # sum over L
        return out