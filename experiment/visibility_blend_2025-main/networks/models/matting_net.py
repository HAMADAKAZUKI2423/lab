import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from .encoders import AlphaNetEncoder
from .film import FilmMlp
from .decoders import FilmDecoder
from .INetwork import INetwork

class MattingNetBase(INetwork):
    
    def __init__(self, in_channels=16, max_channels=128, num_layer = 4):
        super(MattingNetBase, self).__init__()
        #encoder
        #1 conv3x3-relu 3->16 stride=1 1/1
        #2 conv3x3-bn-relu 16->32 stride=2 1/2
        #3 conv3x3-bn-relu 32->64 stride=2 1/4
        #4 conv3x3-bn-relu 64->128 stride=2 1/8

        #5 conv3x3-bn-relu 128->128 stride=2 1/16
        #6 conv3x3-bn-relu 128->128 stride=2 1/32
        #7 conv3x3-bn-relu 128->128 stride=2 1/64

        self.feature_channels = []
        self.feature_channels.append(in_channels)
        out_ch=in_channels
        for i in range(1,num_layer):
            in_ch = out_ch
            if in_ch*2>max_channels:
                out_ch = in_ch
            else:
                out_ch = in_ch*2
            self.feature_channels.append(out_ch)

        print(f"self.feature_channels: {self.feature_channels}")
        self.backbone = AlphaNetEncoder(6, self.feature_channels, depthwise=True)
        self.vis_backbone = AlphaNetEncoder(1, self.feature_channels, depthwise=True)
        self.aspp = ASPP(self.feature_channels[-1], [2, 4, 8],max_channels)
        self.film_mlp = FilmMlp(self.feature_channels,[*self.feature_channels[:-1],max_channels])
        self.decoder = FilmDecoder([*self.feature_channels[:-1],max_channels],1)
        self.binding = nn.Sequential(
            nn.Conv2d(self.feature_channels[0], 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2, tv):
        x = torch.cat([x1,x2],dim=1)
        x_code = self.backbone(x)
        x_code[-1] = self.aspp(x_code[-1])
        tv = tv * 2.0-1.0
        vis_code = self.vis_backbone(tv)
        s_code, m_code = self.film_mlp(vis_code)
        _y = self.decoder(x_code,s_code, m_code)
        _y = self.binding(_y)
        return _y