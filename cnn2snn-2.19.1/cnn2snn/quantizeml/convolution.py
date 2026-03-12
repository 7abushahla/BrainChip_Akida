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
"""Functions to convert QuantizedConv2D to Akida.
"""

from quantizeml.layers import (QuantizedConv2D, QuantizedReLU, QuantizedGlobalAveragePooling2D,
                               QuantizedMaxPool2D, QuantizedActivation)
from akida import Convolutional, Conv2D

from ..akida_versions import AkidaVersion
from .pooling import (max_pool_param_checks, gap_params_checks,
                      set_gap_variables_v1)
from .outputs import set_output_v1_variables
from .padding import check_conv_and_max_pool_compatibility
from .blocks import get_block_out_quantizer
from .block_converter import BlockConverter, register_conversion_patterns
from .conv_common import (parse_conv_block_v1, get_layer_by_type,
                          parse_conv_block_v2, set_conv_variables_v1, set_conv_variables_v2)

__all__ = ["ConvBlockConverterV1", "ConvBlockConverterV2"]

_PATTERNS_V1 = [(QuantizedConv2D,), (QuantizedConv2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedGlobalAveragePooling2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedGlobalAveragePooling2D)]

_PATTERNS_V2 = [(QuantizedConv2D,), (QuantizedConv2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedReLU, QuantizedGlobalAveragePooling2D),
                (QuantizedConv2D, QuantizedGlobalAveragePooling2D),
                (QuantizedConv2D, QuantizedActivation),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedActivation),
                (QuantizedConv2D, QuantizedActivation, QuantizedGlobalAveragePooling2D)]


def convert_conv_block_v1(model_ak, block, inbound_layers_ak):
    """Converts a convolutional block into an akida v1 Convolutional layer.

    The expected sequence is:

    - QuantizedConv2D,
    - QuantizedMaxPooling2D or QuantizedGlobalAveragePooling2D (optional),
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """

    conv = block[0]

    # Evaluate the convolutional block layers parameters
    conv_params = parse_conv_block_v1(block)

    # Add layer to the model to build its internal variables
    conv_ak = Convolutional(**conv_params)
    model_ak.add(conv_ak, inbound_layers_ak)

    # Set base variables
    set_conv_variables_v1(conv_ak, conv)

    # Check if we have GAP
    pool_layer = get_layer_by_type(block, QuantizedGlobalAveragePooling2D)
    if pool_layer:
        set_gap_variables_v1(conv_ak, pool_layer)

    # Get out_quantizer of the block.
    out_quantizer = get_block_out_quantizer(block)

    if out_quantizer:
        set_output_v1_variables(conv_ak, out_quantizer)


def convert_conv_block_v2(model_ak, block, inbound_layers_ak):
    """Converts a convolutional block into an akida v2 Conv2D layer.

    The expected sequence is:

    - QuantizedConv2D,
    - QuantizedMaxPooling2D (optional),
    - QuantizedReLU (optional).

    or:
    - QuantizedConv2D,
    - QuantizedReLU (optional),
    - QuantizedGlobalAveragePooling2D (optional).

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the convolutional block layers parameters
    conv_params = parse_conv_block_v2(block)

    # Add layer to the model to build its internal variables
    conv_ak = Conv2D(**conv_params)
    model_ak.add(conv_ak, inbound_layers_ak)

    # Set base variables
    set_conv_variables_v2(conv_ak, block)


class ConvBlockConverterV1(BlockConverter):
    """Main class that should be used to check if the conv block is compatible to an Akida v1
    conversion and provides a method to convert it in an equivalent Akida Convolutional layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._conv_additional_checks()

    def _conv_additional_checks(self):
        conv = self._block[0]

        # The only weight bitwidth supported is [2, 4]
        weight_bits = conv.weight_quantizer.bitwidth
        if weight_bits not in [1, 2, 4]:
            raise ValueError("Convolutional layer can only handle weights with 1, 2 or 4 bits."
                             f" Received: {weight_bits}.")
        pool_layer = get_layer_by_type(self._block, (QuantizedGlobalAveragePooling2D,
                                                     QuantizedMaxPool2D))
        if isinstance(pool_layer, QuantizedMaxPool2D):
            check_conv_and_max_pool_compatibility(conv, pool_layer)
        elif isinstance(pool_layer, QuantizedGlobalAveragePooling2D):
            gap_params_checks(pool_layer)

    def convert(self, model_ak, inbounds):
        convert_conv_block_v1(model_ak, self._block, inbounds)


class ConvBlockConverterV2(BlockConverter):
    """Main class that should be used to check if the conv block is compatible to an Akida v2
    conversion and provides a method to convert it in an equivalent Akida Conv2D layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._conv_additional_checks()

    def _conv_additional_checks(self):
        conv = self._block[0]
        pool_layer = get_layer_by_type(self._block, QuantizedMaxPool2D)
        if pool_layer:
            check_conv_and_max_pool_compatibility(conv, pool_layer)
            max_pool_param_checks(pool_layer)

        # The only weight bitwidth supported is [4, 8]
        weight_bits = conv.weight_quantizer.bitwidth
        if weight_bits not in [4, 8]:
            raise ValueError("Conv2D layer can only handle weights with 4 or 8 bits."
                             f" Received: {weight_bits}.")

        # Make sure the Conv2D layers spatial params are square
        assert conv.kernel_size[0] == conv.kernel_size[1], ("Conv2D handle only"
                                                            "square kernels")
        assert conv.strides[0] == conv.strides[1], ("Conv2D stride should be"
                                                    "the same for both dimensions")

    def convert(self, model_ak, inbounds):
        convert_conv_block_v2(model_ak, self._block, inbounds)


# Register the valid conv block patterns for Akida v1 and v2
register_conversion_patterns(AkidaVersion.v1, _PATTERNS_V1, ConvBlockConverterV1)
register_conversion_patterns(AkidaVersion.v2, _PATTERNS_V2, ConvBlockConverterV2)
