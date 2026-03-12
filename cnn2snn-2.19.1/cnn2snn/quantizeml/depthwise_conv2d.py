#!/usr/bin/env python
# ******************************************************************************
# Copyright 2023 Brainchip Holdings Ltd.
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
"""Functions to convert QuantizedDepthwiseConv2D to Akida.
"""
from quantizeml.layers import (QuantizedDepthwiseConv2D, QuantizedReLU, QuantizedMaxPool2D,
                               QuantizedActivation)
from akida import DepthwiseConv2D

from ..akida_versions import AkidaVersion
from .padding import check_conv_and_max_pool_compatibility
from .pooling import max_pool_param_checks
from .conv_common import get_layer_by_type, parse_conv_block_v2, set_conv_variables_v2
from .block_converter import BlockConverter, register_conversion_patterns

__all__ = ["DepthwiseConvBlockConverter"]

_PATTERNS = [(QuantizedDepthwiseConv2D,), (QuantizedDepthwiseConv2D, QuantizedReLU),
             (QuantizedDepthwiseConv2D, QuantizedMaxPool2D, QuantizedReLU),
             (QuantizedDepthwiseConv2D, QuantizedActivation),
             (QuantizedDepthwiseConv2D, QuantizedMaxPool2D, QuantizedActivation)]


def convert_depthwise_conv_block(model_ak, block, inbound_layers_ak):
    """Converts a depthwise convolutional block into an akida v2 DepthwiseConv2D layer.

    The expected sequence is:

    - QuantizedDepthwiseConv2D,
    - QuantizedMaxPool2D (optional),
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the depthwise convolutional layer parameters
    conv_params = parse_conv_block_v2(block, depthwise=True)

    # Create Akida layer
    dw_conv_ak = DepthwiseConv2D(**conv_params)
    # Add layer to the model to build its internal variables
    model_ak.add(dw_conv_ak, inbound_layers_ak)

    # Set base variables
    set_conv_variables_v2(dw_conv_ak, block)


class DepthwiseConvBlockConverter(BlockConverter):
    """Main class that should be used to check if the depthwise conv block is compatible to an
    Akida v2 conversion and provides a method to convert it in an equivalent Akida DepthwiseConv2D
    layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._depthwise_conv_additional_checks()

    def _depthwise_conv_additional_checks(self):
        depth_conv = self._block[0]
        # Make sure the DepthwiseConv2D kernel size and stride params are square
        assert depth_conv.kernel_size[0] == depth_conv.kernel_size[1], (
            "DepthwiseConv2D kernel should be square")
        assert depth_conv.strides[0] == depth_conv.strides[1], (
            "DepthwiseConv2D strides should be the same on both dimensions")

        # The only weight bitwidth supported is [4, 8]
        weight_bits = depth_conv.weight_quantizer.bitwidth
        assert weight_bits in [4, 8], ("DepthwiseConv2D layer can only handle weights"
                                       f" with 4 or 8 bits. Received: {weight_bits}.")

        # Check optional pooling compatibility
        pool_layer = get_layer_by_type(self._block, QuantizedMaxPool2D)
        if pool_layer:
            check_conv_and_max_pool_compatibility(depth_conv, pool_layer)
            max_pool_param_checks(pool_layer)

    def convert(self, model_ak, inbounds):
        convert_depthwise_conv_block(model_ak, self._block, inbounds)


# Register the valid depthwise conv block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, DepthwiseConvBlockConverter)
