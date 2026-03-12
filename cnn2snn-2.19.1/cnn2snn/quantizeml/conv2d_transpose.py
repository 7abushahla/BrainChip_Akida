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
"""Functions to convert QuantizedConv2DTranspose to Akida.
"""

from quantizeml.layers import QuantizedConv2DTranspose, QuantizedReLU, QuantizedActivation
from akida import Conv2DTranspose
from ..akida_versions import AkidaVersion
from .block_converter import BlockConverter, register_conversion_patterns
from .conv_common import set_conv_variables_v2, parse_conv_block_v2

__all__ = ["ConvTransposeBlockConverter"]

_PATTERNS = [(QuantizedConv2DTranspose,), (QuantizedConv2DTranspose, QuantizedReLU),
             (QuantizedConv2DTranspose, QuantizedActivation)]


def convert_conv_transpose_block(model_ak, block, inbound_layers_ak):
    """Converts a convolutional transpose block into an akida Conv2DTranspose layer.

    The expected sequence is:

    - QuantizedConv2DTranspose,
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the convolutional transpose layer parameters
    conv_transpose_params = parse_conv_block_v2(block, transpose=True)

    # Create Akida layer
    conv_transpose_ak = Conv2DTranspose(**conv_transpose_params)
    # Add layer to the model to build its internal variables
    model_ak.add(conv_transpose_ak, inbound_layers_ak)

    # Set base variables
    set_conv_variables_v2(conv_transpose_ak, block, flip=False)


class ConvTransposeBlockConverter(BlockConverter):
    """Main class that should be used to check if the conv transpose block is compatible to an
    Akida v2 conversion and provides a method to convert it in an equivalent Akida Conv2DTranspose
    layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._conv_transpose_additional_checks()

    def _conv_transpose_additional_checks(self):
        conv_transpose = self._block[0]

        assert conv_transpose.kernel_size[0] == conv_transpose.kernel_size[1], (
            "QuantizedConv2DTranspose kernel should be square")
        assert conv_transpose.kernel_size[0] in (3, 4)
        assert conv_transpose.strides[0] == conv_transpose.strides[1], (
            "QuantizedConv2DTranspose strides should be the same on both dimensions.")

        assert conv_transpose.strides[0] == 2, ("QuantizedConv2DTranspose handles only stride 2")
        # Padding value must be built in constructor
        assert conv_transpose.padding == "same", ("QuantizedConv2DTranspose handles only 'same' "
                                                  "padding")
        # The only weight bitwidth supported is [4, 8]
        weight_bits = conv_transpose.weight_quantizer.bitwidth
        assert weight_bits in [4, 8], ("QuantizedConv2DTranspose layer can only handle weights"
                                       f" with 4 or 8 bits. Received: {weight_bits}.")

    def convert(self, model_ak, inbounds):
        convert_conv_transpose_block(model_ak, self._block, inbounds)


# Register the valid conv2d transpose block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, ConvTransposeBlockConverter)
