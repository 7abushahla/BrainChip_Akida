#!/usr/bin/env python
# ******************************************************************************
# Copyright 2024 Brainchip Holdings Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
__all__ = ["DepthwiseConv2DTransposeOnnxConverter"]

import akida

from .conv2d_transpose import Conv2DTransposeOnnxConverter
from .conv_commons import parse_convolutional_parameters
from .register import register_onnx_converter_target


@register_onnx_converter_target("QuantizedDepthwise2DTranspose")
class DepthwiseConv2DTransposeOnnxConverter(Conv2DTransposeOnnxConverter):
    """Convert QuantizedDepthwise2DTranspose type node into an akida.DepthwiseConv2DTranspose.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model that the node is.
    """

    def _parse_akida_layer(self):
        layer_params = parse_convolutional_parameters(self, depthwise=True, transpose=True)
        return akida.DepthwiseConv2DTranspose(**layer_params)
