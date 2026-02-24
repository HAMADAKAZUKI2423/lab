from torch import nn
from torchvision.models import MobileNetV2
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops.misc import Conv2dNormActivation 

class AlphaNetEncoder(nn.Module):
    
    def __init__(self, image_channels, feature_channels, depthwise = False):
        super().__init__()
        
        self.features = nn.ModuleList()
        if depthwise:
            for i in range(len(feature_channels)):
                if i == 0:
                    self.features.append(nn.Sequential(
                        depthwise_separable_conv(image_channels, feature_channels[0], kernel_size=3, stride=1, padding=1),
                        nn.ReLU(True)
                    ))
                else:
                    self.features.append(nn.Sequential(
                        depthwise_separable_conv(feature_channels[i-1], feature_channels[i], kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(feature_channels[i]),
                        nn.ReLU(True)
                    ))
        else:
            for i in range(len(feature_channels)):
                if i == 0:
                    self.features.append(nn.Sequential(
                        nn.Conv2d(image_channels, feature_channels[0], kernel_size=3, stride=1, padding=1),
                        nn.ReLU(True)
                    ))
                else:
                    self.features.append(nn.Sequential(
                        nn.Conv2d(feature_channels[i-1], feature_channels[i], kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(feature_channels[i]),
                        nn.ReLU(True)
                    ))
        
    def forward(self, x):
        x_code = []
        for i,enc in enumerate(self.features):
            if i==0:
                x_code.append(enc(x))
            else:
                x_code.append(enc(x_code[-1]))

        return x_code

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 2, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride = stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method 
    additionally returns the feature maps at all resolutions for decoder's use.
    """
    
    def __init__(self, in_channels, norm_layer=None):
        super().__init__()
        
        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
       
        # Remove last block
        self.features = self.features[:-1]
        
        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)
        
        # Delete classifier
        del self.classifier
        
    def forward(self, x):
        x0 = x  # 1/1
        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x  # 1/2
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x  # 1/4
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x  # 1/8
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0

class InvertedResidualEncoder(nn.Module):
    
    def __init__(self, image_channels, inverted_residual_setting):
        super().__init__()
        
        self.features = nn.ModuleList()
        norm_layer = nn.BatchNorm2d

        for j, (t, c, n, s) in enumerate(inverted_residual_setting):
            for i in range(n):
                stride = s if i == 0 else 1
                if j == 0:
                    self.features.append(Conv2dNormActivation(image_channels, c, stride, norm_layer=norm_layer, activation_layer=nn.ReLU6))
                else:
                    self.features.append(InvertedResidual(input_channel, c, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = c
        
    def forward(self, x):
        x_code = []
        for i,enc in enumerate(self.features):
            if i==0:
                x_code.append(enc(x))
            else:
                x_code.append(enc(x_code[-1]))

        return x_code
    