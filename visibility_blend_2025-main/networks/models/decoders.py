import torch
from torch import nn
import torch.nn.functional as F
from .film import BasicBlockFilm

class SimpleDecoder(nn.Module):
    def __init__(self, channels, feature_channels):
        super().__init__()
        
        self.initInterpolate = channels[0] != 0

        self.features = nn.ModuleList()
        num_layer = len(feature_channels)

        for i in range(0, num_layer):
            in_ch =  channels[i] + feature_channels[i]
            out_ch = channels[i+1]
            if i != num_layer-1:
                self.features.append(nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(True)
                        ))
            else:
                self.features.append(nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, padding=1)
                        ))
            
    def forward(self, x_code):
        for i,dec in enumerate(self.features):
            if self.initInterpolate:
                if i==0:
                    _y = F.interpolate(x_code[-1-i], size=x_code[-2-i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    _y = F.interpolate(_y, size=x_code[-2-i].shape[2:], mode='bilinear', align_corners=False)
                _y = torch.cat([_y,x_code[-2-i]],dim=1)
            else:
                if i==0:
                    _y = x_code[-1-i]
                else:
                    _y = F.interpolate(_y, size=x_code[-1-i].shape[2:], mode='bilinear', align_corners=False)
                    _y = torch.cat([_y,x_code[-1-i]],dim=1)
            _y = dec(_y)

        return _y

class BasicBlockFilm(nn.Module):
    
    def __init__(self, in_planes, planes, use_pixelshuffle=True, use_relu=True):
        super(BasicBlockFilm, self).__init__()

        self.use_pixelshuffle = use_pixelshuffle
        self.use_relu = use_relu

        if self.use_pixelshuffle:
            self.wide_layer = nn.Conv2d(in_planes, in_planes*4,
                                        kernel_size=1, stride=1,
                                        padding=0)
            self.wide_act = nn.LeakyReLU(0.2, True)
            self.shuffle = nn.PixelShuffle(2)
            self.up_conv_layer = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
            
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        

    def forward(self, x, sigma=None, mu=None):
        if self.use_pixelshuffle:
            out = self.bn1(self.up_conv_layer(self.shuffle(self.wide_act(self.wide_layer(x)))))
        else:
            out = self.bn1(self.conv1(x))
        if sigma is not None and mu is not None:
            out = out * sigma + mu
        if self.use_relu:
            out = F.relu(out)

        return out

class FilmDecoder2(nn.Module):
    def __init__(self, channels, feature_channels):
        super().__init__()

        self.initInterpolate = channels[0] != 0
        
        self.features = nn.ModuleList()
        num_layer = len(feature_channels)

        for i in range(0, num_layer):
            in_ch =  channels[i] + feature_channels[i]
            out_ch = channels[i+1]
            if i != num_layer-1:
                self.features.append(BasicBlockFilm(in_ch, out_ch))
            else:
                self.features.append(BasicBlockFilm(in_ch, out_ch,use_pixelshuffle=False))

    def forward(self, x_code,s_code,m_code):
        for i,dec in enumerate(self.features):
            if self.initInterpolate:
                if i==0:
                    _y = F.interpolate(x_code[-1-i], size=x_code[-2-i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    _y = F.interpolate(_y, size=x_code[-2-i].shape[2:], mode='bilinear', align_corners=False)
                _y = torch.cat([_y,x_code[-2-i]],dim=1)
            else:
                if i==0:
                    _y = x_code[-1-i]
                else:
                    _y = F.interpolate(_y, size=x_code[-1-i].shape[2:], mode='bilinear', align_corners=False)
                    _y = torch.cat([_y,x_code[-1-i]],dim=1)
            
            _y = dec(_y,s_code[-1-i],m_code[-1-i])

        return _y


class FilmDecoder(nn.Module):
    def __init__(self, feature_channels, num_inputs):
        super().__init__()
        
        self.features = nn.ModuleList()
        num_layer = len(feature_channels)

        for i in range(num_layer-1,0,-1):
            if i==num_layer-1:
                in_ch = feature_channels[i]*num_inputs
                out_ch = feature_channels[i-1]
            else:
                in_ch = feature_channels[i]*(num_inputs+1)
                out_ch = feature_channels[i-1]
            self.features.append(BasicBlockFilm(in_ch, out_ch))
        
        self.features.append(BasicBlockFilm(feature_channels[0]*(num_inputs+1), feature_channels[0],use_pixelshuffle=False))
        
    def forward(self, x_code,s_code,m_code):
        for i,dec in enumerate(self.features):
            if i==0:
                _y = dec(x_code[-1-i],s_code[-1-i],m_code[-1-i])
            else:
                _y = dec(torch.cat([x_code[-1-i],_y],dim=1),s_code[-1-i],m_code[-1-i])

        return _y
