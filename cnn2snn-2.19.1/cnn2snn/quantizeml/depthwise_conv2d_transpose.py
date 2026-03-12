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
"""Functions to convert QuantizedDepthwiseConv2DTranspose to Akida.
"""

from quantizeml.layers import QuantizedDepthwiseConv2DTranspose, QuantizedReLU, QuantizedActivation
from akida import DepthwiseConv2DTranspose

from ..akida_versions import AkidaVersion
from .block_converter import BlockConverter, register_conversion_patterns
from .conv_common import parse_conv_block_v2, set_conv_variables_v2

__all__ = ["DepthwiseConvTransposeBlockConverter"]

_PATTERNS = [(QuantizedDepthwiseConv2DTranspose,),
             (QuantizedDepthwiseConv2DTranspose, QuantizedReLU),
             (QuantizedDepthwiseConv2DTranspose, QuantizedActivation)]


def convert_depthwise_conv2d_transpose_block(model_ak, block, inbound_layers_ak):
    """Converts a depthwise convolutional transpose block into an akida v2 DepthwiseConv2DTranspose
    layer.

    The expected sequence is:

    - QuantizedDepthwiseConv2DTranspose,
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the depthwise convolutional transpose layer parameters
    conv_params = parse_conv_block_v2(block, depthwise=True, transpose=True)

    # Create Akida layer
    dw_conv_ak = DepthwiseConv2DTranspose(**conv_params)
    # Add layer to the model to build its internal variables
    model_ak.add(dw_conv_ak, inbound_layers_ak)

    # Set base variables
    set_conv_variables_v2(dw_conv_ak, block, flip=False)


class DepthwiseConvTransposeBlockConverter(BlockConverter):
    """Main class that should be used to check if the depthwise conv transpose block is compatible
    to an Akida v2 conversion and provides a method to convert it in an equivalent Akida
    DepthwiseConv2DTranspose layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._depthwise_conv_additional_checks()

    def _depthwise_conv_additional_checks(self):
        depth_transpose = self._block[0]

        # Make sure the DepthwiseConv2DTranspose kernel size and stride params are square
        assert depth_transpose.kernel_size[0] == depth_transpose.kernel_size[1], (
            "DepthwiseConv2DTranspose kernel should be square")
        assert depth_transpose.strides[0] == depth_transpose.strides[1], (
            "DepthwiseConv2DTranspose strides should be the same on both dimension")
        assert depth_transpose.kernel_size[0] in [3, 4], (
            "DepthwiseConv2DTranspose supports kernel size of 3 or 4 only")
        assert depth_transpose.strides[0] == 2, "DepthwiseConv2DTranspose supports stride 2 only"

        # Padding value must be same
        assert depth_transpose.padding == "same", ("DepthwiseConv2DTranspose handles only 'same'"
                                                   " padding")

        # The only weight bitwidth supported is [4, 8]
        weight_bits = depth_transpose.weight_quantizer.bitwidth
        assert weight_bits in [4, 8], ("DepthwiseConv2DTranspose layer can only handle weights"
                                       f" with 4 or 8 bits. Received: {weight_bits}.")

    def convert(self, model_ak, inbounds):
        convert_depthwise_conv2d_transpose_block(model_ak, self._block, inbounds)


# Register the valid depthwise conv transpose block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, DepthwiseConvTransposeBlockConverter)
