import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import Conv2dNormActivation
from typing import Optional, Callable, List, Sequence

class InvertedResidualReflection(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                    norm_layer(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
            )
        layers.extend(
            [
                # dw with reflection padding
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, bias=False),
                    norm_layer(hidden_dim),
                    nn.ReLU6(inplace=True)
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ASPPConvReflection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class ASPPPoolingReflection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Sequential(
                nn.ReflectionPad2d(8),  # Padding to handle stride
                nn.AvgPool2d(kernel_size=16, stride=8),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
# class ASPPPoolingReflection(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super().__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPPReflection(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            modules.append(ASPPConvReflection(in_channels, out_channels, rate))

        modules.append(ASPPPoolingReflection(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class SimpleDecoderReflection(nn.Module):
    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channels[0] + channels[0], channels[1], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channels[1] + channels[1], channels[2], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channels[2] + channels[2], channels[3], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channels[3] + channels[3], channels[4], kernel_size=3, padding=0)
        )

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        return x


class InvertedResidualEncoderReflection(nn.Module):
    def __init__(self, image_channels, inverted_residual_setting):
        super().__init__()

        self.features = nn.ModuleList()
        norm_layer = nn.BatchNorm2d

        # for j, (t, c, n, s) in enumerate(inverted_residual_setting):
        #     for i in range(n):
        #         stride = s if i == 0 else 1
        #         if j == 0:
        #             self.features.append(Conv2dNormActivation(image_channels, c, stride, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        #         else:
        #             self.features.append(InvertedResidualReflection(image_channels, c, stride, expand_ratio=t, norm_layer=norm_layer))
        #         image_channels = c
        input_channel = image_channels
        for j, (t, c, n, s) in enumerate(inverted_residual_setting):
            for i in range(n):
                stride = s if i == 0 else 1
                if j == 0 and i == 0:
                    self.features.append(Conv2dNormActivation(input_channel, c, stride, norm_layer=norm_layer, activation_layer=nn.ReLU6))
                else:
                    self.features.append(InvertedResidualReflection(input_channel, c, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = c

    def forward(self, x):
        x_code = []
        for i, enc in enumerate(self.features):
            if i == 0:
                x_code.append(enc(x))
            else:
                x_code.append(enc(x_code[-1]))

        return x_code

# Update TestNet


# Test cases
if __name__ == "__main__":
    
    class TestNet(nn.Module):
        def __init__(self, out_channels=1, mode=0):
            super(TestNet, self).__init__()
            self.mode = mode

            if mode == 0:
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

        def forward(self, x1, x2, tv):
            tv = tv * 2.0 - 1.0
            x = torch.cat([x1, x2, tv], dim=1)
            x = self.backbone(x)
            if self.mode in [0]:
                x_aspp = self.aspp(x[-1])
                #x = self.decoder([*x, x_aspp])
                x = self.decoder(x_aspp, x[-1], x[-2], x[-3], x[-4])
            x = self.binding(x)
            return x

    def test_testnet():
        # Define dummy inputs
        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)
        tv = torch.randn(1, 1, 128, 128)

        # Initialize TestNet
        model = TestNet(out_channels=1, mode=0)
        model.eval()  # Set to evaluation mode

        # Forward pass
        output = model(x1, x2, tv)

        # Assertions
        assert output.shape == (1, 1, 128, 128), f"Unexpected output shape: {output.shape}"
        print("TestNet test passed!")

    # Run tests
    test_testnet()
