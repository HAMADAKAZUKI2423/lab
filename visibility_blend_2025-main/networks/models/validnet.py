import torch
from torch import nn
from torch.nn.functional import interpolate
from typing import Sequence
from torchvision.ops.misc import Conv2dNormActivation

class CustomInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(CustomInvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(
                Conv2dNormActivation(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
                )
            )
        # Depthwise
        layers.append(
            Conv2dNormActivation(
                hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # Pointwise-linear
        layers.extend([
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ValidPaddingInvertedResidualEncoder(nn.Module):
    def __init__(self, image_channels, inverted_residual_setting):
        super().__init__()
        self.features = nn.ModuleList()
        norm_layer = nn.BatchNorm2d

        input_channel = image_channels
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(CustomInvertedResidual(input_channel, c, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = c

    def forward(self, x):
        x_code = []
        for layer in self.features:
            x = layer(x)
            x_code.append(x)
        return x_code

class ValidPaddingASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=0, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)

class ValidPaddingASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ValidPaddingASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))# これを一定範囲のpoolingにする

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return interpolate(x, size=size, mode="bilinear", align_corners=False)

class ValidPaddingDecoder(nn.Module):
    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=0)
        self.relu = nn.ReLU(True)

    def forward(self, x4, x3, x2, x1, x0):
        x = interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        return x

class TestNet(nn.Module):
    def __init__(self, out_channels=1, mode=0):
        super(TestNet, self).__init__()
        self.mode = mode

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 1, 2],
            [6, 32, 1, 2],
            [6, 64, 1, 2]
        ]

        self.backbone = ValidPaddingInvertedResidualEncoder(7, inverted_residual_setting)
        # self.aspp = ValidPaddingASPP(inverted_residual_setting[-1][1], [3, 6, 9], 128)
        self.aspp = ValidPaddingASPP(inverted_residual_setting[-1][1], [2, 4, 6], 128)
        self.decoder = ValidPaddingDecoder([128, 32, 24, 16, out_channels], [64, 32, 24, 16])
        self.binding = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, tv):
        tv = tv * 2.0 - 1.0
        x = torch.cat([x1, x2, tv], dim=1)
        x = self.backbone(x)
        x_aspp = self.aspp(x[-1])
        x = self.decoder([*x, x_aspp])
        x = self.binding(x)
        return x

# Test script
if __name__ == "__main__":
    print("test started")
    def test_TestNet():
        # Create the model
        model = TestNet(out_channels=1, mode=0)
        model.eval()  # Set to evaluation mode

        # Initialize min_size as None
        min_size = None

        # Determine the minimum input size
        for size in range(97, 128):  # Arbitrary upper bound for testing
            try:
                x1 = torch.randn(1, 3, size, size)
                x2 = torch.randn(1, 3, size, size)
                tv = torch.randn(1, 1, size, size)
                with torch.no_grad():
                    output = model(x1, x2, tv)
                print(f"Minimum input size for mode=0: {size}")
                min_size = size
                break
            except Exception as e:
                print(f"Size {size} failed: {e}")

        # Validate that min_size was found
        if min_size is None:
            print("No valid minimum size found.")
            return

        # Test with a valid size
        x1 = torch.randn(1, 3, min_size, min_size)
        x2 = torch.randn(1, 3, min_size, min_size)
        tv = torch.randn(1, 1, min_size, min_size)

        with torch.no_grad():
            output = model(x1, x2, tv)
            assert output.shape == (1, 1, min_size, min_size), f"Unexpected output shape: {output.shape}"
            print("Test passed: output shape is correct.")

    test_TestNet()