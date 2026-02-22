import torch.nn.functional as F
from torch import nn

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

class FilmMlp2(nn.Module):
    
    def __init__(self, feature_channels, out_channels):
        super(FilmMlp2, self).__init__()

        self.sigma_decoder = nn.ModuleList()
        self.mu_decoder = nn.ModuleList()

        mlp_ksize = 1
        for i in range(0,len(feature_channels)):
            self.sigma_decoder.append(self.__mlp(mlp_ksize,feature_channels[i],feature_channels[i]//2,out_channels[i]))
            self.mu_decoder.append(self.__mlp(mlp_ksize,feature_channels[i],feature_channels[0]//2,out_channels[i]))
        
    def __mlp(self, ksize, input, hidden, output):
        pad=(ksize-1)//2

        if hidden==0:
            layer = [
                nn.Conv2d(input, output, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True)
            ]

        else:
            layer = [
                nn.Conv2d(input, hidden, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, output, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True)
            ]

        return nn.Sequential(*layer)
    
    def forward(self, vis_code):
        s_code = []
        m_code = []

        for i in range(0,len(vis_code)):
            s_code.append(self.sigma_decoder[i](vis_code[i]))
            m_code.append(self.mu_decoder[i](vis_code[i]))
        
        return s_code, m_code

class FilmMlp(nn.Module):
    
    def __init__(self, feature_channels, out_channels):
        super(FilmMlp, self).__init__()

        self.sigma_decoder = nn.ModuleList()
        self.mu_decoder = nn.ModuleList()

        mlp_ksize = 1

        self.sigma_decoder.append(self.__mlp(mlp_ksize,feature_channels[0],feature_channels[0]//2,out_channels[0]))
        self.mu_decoder.append(self.__mlp(mlp_ksize,feature_channels[0],feature_channels[0]//2,out_channels[0]))

        for i in range(0,len(feature_channels)-1):
            self.sigma_decoder.append(self.__mlp(mlp_ksize,feature_channels[i],feature_channels[i]//2,out_channels[i]))
            self.mu_decoder.append(self.__mlp(mlp_ksize,feature_channels[i],feature_channels[0]//2,out_channels[i]))
        
    def __mlp(self, ksize, input, hidden, output):
        pad=(ksize-1)//2

        if hidden==0:
            layer = [
                nn.Conv2d(input, output, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True)
            ]

        else:
            layer = [
                nn.Conv2d(input, hidden, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, output, (ksize,ksize), stride = 1, padding =(pad,pad)),
                nn.ReLU(inplace=True)
            ]

        return nn.Sequential(*layer)
    
    def forward(self, vis_code):
        s_code = []
        m_code = []

        s_code.append(self.sigma_decoder[0](vis_code[0]))
        m_code.append(self.mu_decoder[0](vis_code[0]))
        
        for i in range(0,len(vis_code)-1):
            s_code.append(self.sigma_decoder[i+1](vis_code[i]))
            m_code.append(self.mu_decoder[i+1](vis_code[i]))
        
        return s_code, m_code
