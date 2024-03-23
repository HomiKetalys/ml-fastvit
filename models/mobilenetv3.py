#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union, Sequence, Callable, Any

import torch
import torch.nn as nn
from timm.models import register_model
from torch import Tensor
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, _mobilenet_v3
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation
from torchvision.utils import _log_api_usage_once


def _mobilenet_v3_conf(
        arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "RE", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "RE", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "RE", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "RE", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "RE", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "RE", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "RE", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "RE", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "RE", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "RE", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "RE", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "RE", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "RE", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "RE", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "RE", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, activation=nn.ReLU6,
                                                         scale_activation=nn.Hardsigmoid),
            activation=nn.ReLU6,
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = activation

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 101,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
            activation=nn.ReLU6,
            separation=0,
            separation_scale=2,
            **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()
        _log_api_usage_once(self)
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer, activation=activation))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            activation(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.separation = separation
        self.separation_scale = separation_scale
        self.stage_list = [self.features[0], self.features[1], self.features[2:4], self.features[4:9],
                           self.features[9:]]
        self.stage_out_channels = [inverted_residual_setting[0].input_channels,
                                   inverted_residual_setting[1].input_channels,
                                   inverted_residual_setting[3].input_channels,
                                   inverted_residual_setting[8].input_channels,
                                   inverted_residual_setting[-1].out_channels, ]
        self.export = 0

    def _forward_impl(self, x: Tensor) -> Tensor:
        export = self.export
        if export == 0 or (export > 0 and self.separation == 0):
            if self.separation:
                x = self.seperate(x)
            for i, stg in enumerate(self.stage_list):
                x = stg(x)
                if self.separation == i + 1:
                    x = self.montage(x)
        elif export == 1:
            for i, stg in enumerate(self.stage_list):
                x = stg(x)
                if self.separation == i + 1:
                    return x
        elif export == 2:
            for i, stg in enumerate(self.stage_list):
                if self.separation < i + 1:
                    x = stg(x)
        elif export == 3:
            assert self.separation > 0
            assert self.separation_scale > 1
            bs, ch, h, w = x.shape
            x_list = []
            for r in range(0, self.separation_scale):
                for c in range(0, self.separation_scale):
                    x_list.append(x[:, :,
                                  r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
                                  c * w // self.separation_scale:(c + 1) * w // self.separation_scale
                                  ])
            fuse = False
            for i, stg in enumerate(self.stage_list):
                if fuse:
                    x = stg(x)
                else:
                    for id, x in enumerate(x_list):
                        x = stg(x)
                        x_list[id] = x
                if self.separation == i + 1:
                    xr_list = []
                    for c in range(0, self.separation_scale):
                        xr_list.append(
                            torch.cat(x_list[c * self.separation_scale:(c + 1) * self.separation_scale], dim=3))
                    x = torch.cat(xr_list, dim=2)
                    fuse = True

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def seperate(self, x):
        bs, ch, h, w = x.shape

        # loop crop method
        # x_list = []
        # for r in range(0, self.separation_scale):
        #     for c in range(0, self.separation_scale):
        #         x_list.append(x[:, :,
        #                       r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
        #                       c * w // self.separation_scale:(c + 1) * w // self.separation_scale
        #                       ])
        # x = torch.cat(x_list, dim=0)

        # change dimension method
        x = x.view(bs, ch, self.separation_scale, h // self.separation_scale, self.separation_scale,
                   w // self.separation_scale)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(bs * self.separation_scale ** 2, ch, h // self.separation_scale, w // self.separation_scale)
        return x

    def montage(self, x):
        bs, ch, h, w = x.shape
        # loop crop method
        # x_list = torch.split(x, bs // self.separation_scale ** 2, dim=0)
        # xr_list = []
        # for c in range(0, self.separation_scale):
        #     xr_list.append(torch.cat(x_list[c * self.separation_scale:(c + 1) * self.separation_scale], dim=3))
        # x = torch.cat(xr_list, dim=2)

        # change dimension method
        x = x.view(bs // self.separation_scale ** 2, self.separation_scale, self.separation_scale, ch, h, w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(bs // self.separation_scale ** 2, ch, self.separation_scale * h,
                      self.separation_scale * w)
        return x

    def fuse_norm(self,mean,std):
        w=self.features[0][0].weight.data
        w=w/torch.tensor(std)[None,:,None,None]
        self.features[0][0].weight.data=w
        b=torch.sum(-torch.tensor(mean)[None,:,None,None]*w,dim=(1,2,3))
        if self.features[0][0].bias is not None:
            self.features[0][0].bias.data=self.features[0][0].bias.data+b
        else:
            self.features[0][0].bias=nn.Parameter(b,requires_grad=True)






def _mobilenet_v3(
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        **kwargs: Any,
) -> MobileNetV3:
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    return model


@register_model
def mobilenet_v3_small_0_75(**kwargs):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", width_mult=0.75, **kwargs)

    return _mobilenet_v3(inverted_residual_setting, last_channel, **kwargs)
