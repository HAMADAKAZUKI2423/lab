import torch
from torch import nn
from torchvision.models.segmentation.deeplabv3 import ASPP
from .encoders import MobileNetV2Encoder
from .decoder import Decoder
from .resnet import ResNetEncoder
from .INetwork import INetwork

class MobileV2DeeplabV3Net(INetwork):
    
    def __init__(self, in_channels=3*2+1, out_channels=1):
        super(MobileV2DeeplabV3Net, self).__init__()
        self.backbone = MobileNetV2Encoder(in_channels)
        self.aspp = ASPP(320, [3, 6, 9])
        self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])
        self.binding = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2, tv):
        tv = tv * 2.0-1.0
        x = torch.cat([x1,x2,tv],dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        x = self.binding(x)
        return x

class Res101DeeplabV3Net(INetwork):
    
    def __init__(self, in_channels=3*2+1, out_channels=1):
        super(Res101DeeplabV3Net, self).__init__()
        self.backbone = ResNetEncoder(in_channels, variant='resnet101')
        self.aspp = ASPP(2048, [3, 6, 9])
        self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        self.binding = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2, tv):
        tv = tv * 2.0-1.0
        x = torch.cat([x1,x2,tv],dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        x = self.binding(x)
        return x

class MobileV2DeeplabV3Net_Scalar(INetwork):
    
    def __init__(self, out_channels=1):
        super(MobileV2DeeplabV3Net_Scalar, self).__init__(tv_input="scalar")
        self.backbone = MobileNetV2Encoder(6)
        self.aspp = ASPP(320, [3, 6, 9], 255)
        self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, 6])
        self.binding = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2, tv):
        tv = tv * 2.0-1.0
        x = torch.cat([x1,x2],dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        tv_tensor = tv.unsqueeze(-1).unsqueeze(-1).expand(
            tv.shape[0], tv.shape[1], x.shape[2], x.shape[3]
        )
        x = torch.cat([x,tv_tensor],dim=1)

        x = self.decoder(x, *shortcuts)
        x = self.binding(x)
        return x

class Res101DeeplabV3Net_Scalar(INetwork):
    
    def __init__(self, out_channels=1):
        super(Res101DeeplabV3Net_Scalar, self).__init__(tv_input="scalar")
        self.backbone = ResNetEncoder(6, variant='resnet101')
        self.aspp = ASPP(2048, [3, 6, 9], 255)
        self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, 6])
        self.binding = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2, tv):
        tv = tv * 2.0-1.0
        x = torch.cat([x1,x2],dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        tv_tensor = tv.unsqueeze(-1).unsqueeze(-1).expand(
            tv.shape[0], tv.shape[1], x.shape[2], x.shape[3]
        )
        x = torch.cat([x,tv_tensor],dim=1)
        
        x = self.decoder(x, *shortcuts)
        x = self.binding(x)
        return x