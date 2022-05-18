# coding=utf-8
# Copyright 2020 The Learning-to-Prompt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Learning-to-Prompt governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of ResNet V1 in Flax.

"Deep Residual Learning for Image Recognition"
He et al., 2015, [https://arxiv.org/abs/1512.03385]
"""

import functools
from typing import Any, List, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), use_bias=False)
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), use_bias=False)


class ResNetBlock(nn.Module):
    """ResNet block without bottleneck used in ResNet-18 and ResNet-34."""

    filters: int
    norm: Any
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x

        x = Conv3x3(self.filters, strides=self.strides, name="conv1")(x)
        x = self.norm(name="bn1")(x)
        x = nn.relu(x)
        x = Conv3x3(self.filters, name="conv2")(x)
        # Initializing the scale to 0 has been common practice since "Fixup
        # Initialization: Residual Learning Without Normalization" Tengyu et al,
        # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
        x = self.norm(scale_init=nn.initializers.zeros, name="bn2")(x)

        if residual.shape != x.shape:
            residual = Conv1x1(
                self.filters, strides=self.strides, name="proj_conv")(
                residual)
            residual = self.norm(name="proj_bn")(residual)

        x = nn.relu(residual + x)
        return x


class BottleneckResNetBlock(ResNetBlock):
    """Bottleneck ResNet block used in ResNet-50 and larger."""

    @nn.compact
    def __call__(self, x):
        residual = x

        x = Conv1x1(self.filters, name="conv1")(x)
        x = self.norm(name="bn1")(x)
        x = nn.relu(x)
        x = Conv3x3(self.filters, strides=self.strides, name="conv2")(x)
        x = self.norm(name="bn2")(x)
        x = nn.relu(x)
        x = Conv1x1(4 * self.filters, name="conv3")(x)
        # Initializing the scale to 0 has been common practice since "Fixup
        # Initialization: Residual Learning Without Normalization" Tengyu et al,
        # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
        x = self.norm(name="bn3")(x)

        if residual.shape != x.shape:
            residual = Conv1x1(
                4 * self.filters, strides=self.strides, name="proj_conv")(
                residual)
            residual = self.norm(name="proj_bn")(residual)

        x = nn.relu(residual + x)
        return x


class ResNetStage(nn.Module):
    """ResNet stage consistent of multiple ResNet blocks."""

    stage_size: int
    filters: int
    block_cls: Type[ResNetBlock]
    norm: Any
    first_block_strides: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        for i in range(self.stage_size):
            x = self.block_cls(
                filters=self.filters,
                norm=self.norm,
                strides=self.first_block_strides if i == 0 else (1, 1),
                name=f"block{i + 1}")(
                x)
        return x


class ResNet(nn.Module):
    """Construct ResNet V1 with `num_classes` outputs.

    Attributes:
      num_classes: Number of nodes in the final layer.
      block_cls: Class for the blocks. ResNet-50 and larger use
        `BottleneckResNetBlock` (convolutions: 1x1, 3x3, 1x1), ResNet-18 and
          ResNet-34 use `ResNetBlock` without bottleneck (two 3x3 convolutions).
      stage_sizes: List with the number of ResNet blocks in each stage. Number of
        stages can be varied.
      width_factor: Factor applied to the number of filters. The 64 * width_factor
        is the number of filters in the first stage, every consecutive stage
        doubles the number of filters.
      small_input: If True, modify architecture for small inputs like CIFAR.
    """
    num_classes: int
    block_cls: Type[ResNetBlock]
    stage_sizes: List[int]
    width_factor: int = 1
    small_input: bool = False
    train: bool = False
    init_head: str = "zero"

    @nn.compact
    def __call__(self, x):
        """Apply the ResNet to the inputs `x`.

        Args:
          x: Inputs.

        Returns:
          The output head with `num_classes` entries.
        """
        width = 64 * self.width_factor
        norm = functools.partial(
            nn.BatchNorm, use_running_average=not self.train, momentum=0.9)

        # Root block
        if self.small_input:
            x = nn.Conv(
                features=width,
                kernel_size=(3, 3),
                strides=(1, 1),
                use_bias=False,
                name="init_conv")(
                x)
            x = norm(name="init_bn")(x)
            x = nn.relu(x)
        else:
            x = nn.Conv(
                features=width,
                kernel_size=(7, 7),
                strides=(2, 2),
                use_bias=False,
                name="init_conv")(
                x)
            x = norm(name="init_bn")(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # Stages
        for i, stage_size in enumerate(self.stage_sizes):
            x = ResNetStage(
                stage_size,
                filters=width * 2 ** i,
                block_cls=self.block_cls,
                norm=norm,
                first_block_strides=(1, 1) if i == 0 else (2, 2),
                name=f"stage{i + 1}")(
                x)

        # Head
        if self.init_head == "zero":
            head_init = nn.initializers.zeros
        elif self.init_head == "kaiming":
            head_init = nn.initializers.kaiming_uniform
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, kernel_init=head_init, name="head")(x)
        return x


ResNet18 = functools.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet18_he = functools.partial(
    ResNet,
    stage_sizes=[2, 2, 2, 2],
    block_cls=ResNetBlock,
    init_head="kaiming")

ResNet34 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = functools.partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = functools.partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = functools.partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
SmallInput_ResNet18 = functools.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, small_input=True)


def create_model(model_name, config):
    """Creates model partial function."""
    del config
    if model_name == "resnet18":
        model_cls = ResNet18
    elif model_name == "resnet18_he":
        model_cls = ResNet18_he
    elif model_name == "resnet50":
        model_cls = ResNet50
    elif model_name == "resnet18_cifar":
        model_cls = SmallInput_ResNet18
    return model_cls
