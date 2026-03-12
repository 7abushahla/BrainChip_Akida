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
"""Functions to convert QuantizedDepthwiseBufferTempConv to Akida.
"""
from quantizeml.layers import QuantizedDepthwiseBufferTempConv, QuantizedReLU
from akida import DepthwiseBufferTempConv
from ..akida_versions import AkidaVersion
from .btc_common import parse_buf_temp_conv_block, set_buffer_temp_conv_block_variables
from .block_converter import BlockConverter, register_conversion_patterns

__all__ = ["DepthwiseBufferTempConvBlockConverter"]

_PATTERNS = [(QuantizedDepthwiseBufferTempConv,), (QuantizedDepthwiseBufferTempConv, QuantizedReLU)]


def _convert_depth_buf_temp_conv_block(model_ak, block, inbound_layers_ak):
    """Converts a buffer temp conv block into an akida DepthwiseBufferTempConv layer.

    The expected sequence is:

    - QuantizedDepthwiseBufferTempConv,
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the depthwise buffer temporal convolution layer parameters
    btc_layer_params = parse_buf_temp_conv_block(block, depthwise=True)

    # Create Akida layer
    btc_layer_ak = DepthwiseBufferTempConv(**btc_layer_params)
    # Add layer to the model to build its internal variables
    model_ak.add(btc_layer_ak, inbound_layers_ak)

    # Set base variables
    set_buffer_temp_conv_block_variables(btc_layer_ak, block, depthwise=True)


class DepthwiseBufferTempConvBlockConverter(BlockConverter):
    """Main class that should be used to check if the depthwise buffer temp conv block is
    compatible to an Akida v2 conversion and provides a method to convert it in an equivalent
    Akida BufferTempConv layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._depth_buffer_temp_conv_additional_checks()

    def _depth_buffer_temp_conv_additional_checks(self):
        btc_layer = self._block[0]

        # weight bitwidth should be 8-bit maximum
        weight_bits = btc_layer.weight_quantizer.bitwidth
        assert weight_bits <= 8, ("DepthwiseBufferTempConv layer handles weights with"
                                  f" maximum 8 bits. Received: {weight_bits}.")

    def convert(self, model_ak, inbounds):
        _convert_depth_buf_temp_conv_block(model_ak, self._block, inbounds)


# Register the valid depthwise buffer temp conv block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, DepthwiseBufferTempConvBlockConverter)
