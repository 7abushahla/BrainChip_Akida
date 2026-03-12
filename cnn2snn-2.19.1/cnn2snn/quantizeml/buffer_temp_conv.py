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
"""Functions to convert QuantizedBufferTempConv to Akida.
"""
from quantizeml.layers import QuantizedBufferTempConv, QuantizedReLU
from akida import BufferTempConv
from ..akida_versions import AkidaVersion
from .btc_common import parse_buf_temp_conv_block, set_buffer_temp_conv_block_variables
from .block_converter import BlockConverter, register_conversion_patterns

__all__ = ["BufferTempConvBlockConverter"]

_PATTERNS = [(QuantizedBufferTempConv,), (QuantizedBufferTempConv, QuantizedReLU)]


def _convert_buf_temp_conv_block(model_ak, block, inbound_layers_ak):
    """Converts a buffer temp conv block into an akida BufferTempConv layer.

    The expected sequence is:

    - QuantizedBufferTempConv,
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """

    # Evaluate the buffer temporal convolution layer parameters
    btc_layer_params = parse_buf_temp_conv_block(block)

    # Create Akida layer
    btc_layer_ak = BufferTempConv(**btc_layer_params)
    # Add layer to the model to build its internal variables
    model_ak.add(btc_layer_ak, inbound_layers_ak)

    # Set base variables
    set_buffer_temp_conv_block_variables(btc_layer_ak, block)


class BufferTempConvBlockConverter(BlockConverter):
    """Main class that should be used to check if the buffer temp conv block is compatible to an
    Akida v2 conversion and provides a method to convert it in an equivalent Akida BufferTempConv
    layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._buffer_temp_conv_additional_checks()

    def _buffer_temp_conv_additional_checks(self):
        btc_layer = self._block[0]

        # weight bitwidth should be 8bit maximum
        weight_bits = btc_layer.weight_quantizer.bitwidth
        assert weight_bits <= 8, ("BufferTempConv layer handles weights with"
                                  f" maximum 8 bits. Received: {weight_bits}.")

    def convert(self, model_ak, inbounds):
        _convert_buf_temp_conv_block(model_ak, self._block, inbounds)


# Register the valid buffer temp conv block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, BufferTempConvBlockConverter)
