import torch
from torch import nn
from .encoders import InvertedResidualEncoder
from torchvision.models.segmentation.deeplabv3 import ASPP
from .decoders import SimpleDecoder, FilmDecoder2
from .film import FilmMlp2
from .INetwork import INetwork

class ScalarTestNet(INetwork):
    
    def __init__(self, out_channels = 1, mode = 0):
        super(ScalarTestNet, self).__init__(tv_input="scalar")
        self.mode = mode
        
        if mode == 0:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 64, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(6, inverted_residual_setting)
            self.aspp = ASPP(inverted_residual_setting[-1][1], [3, 6, 9], 127)
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
                [6, 63, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(6, inverted_residual_setting)
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

            self.backbone = InvertedResidualEncoder(6, inverted_residual_setting)
            self.aspp = ASPP(inverted_residual_setting[-1][1], [3, 6, 9], 63)
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
                [6, 31, 1, 2]
            ]

            self.backbone = InvertedResidualEncoder(6, inverted_residual_setting)
            self.decoder = SimpleDecoder([0, 16, 12, 8, out_channels], [32, 16, 12, 8])
            self.binding = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, tv: torch.Tensor):
        #x1,x2: NCHW
        #tv: N1
        tv = tv * 2.0-1.0
        if self.mode in [0,1,2,3]:
            x = torch.cat([x1,x2],dim=1)
            x = self.backbone(x)
            if self.mode in [0,2]:
                x_aspp = self.aspp(x[-1])
                tv_tensor = tv.unsqueeze(-1).unsqueeze(-1).expand(
                    tv.shape[0], tv.shape[1], x_aspp.shape[2], x_aspp.shape[3]
                )
                x_aspp = torch.cat([x_aspp,tv_tensor],dim=1)
                x = self.decoder([*x, x_aspp])
            else:
                x_last = x[-1]
                tv_tensor = tv.unsqueeze(-1).unsqueeze(-1).expand(
                    tv.shape[0], tv.shape[1], x_last.shape[2], x_last.shape[3]
                )
                x_last = torch.cat([x_last,tv_tensor],dim=1)
                x[-1] = x_last
                x = self.decoder(x)
        else:
            raise NotImplementedError
        x = self.binding(x)
        return x