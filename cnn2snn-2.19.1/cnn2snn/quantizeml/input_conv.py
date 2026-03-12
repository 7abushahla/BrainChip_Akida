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
"""Functions to convert QuantizedConv2D to Akida InputConvolutional or InputConv2D layer.
"""

from quantizeml.layers import (QuantizedConv2D, QuantizedReLU, QuantizedMaxPool2D,
                               QuantizedActivation)
from akida import InputConvolutional, InputConv2D

from ..akida_versions import AkidaVersion
from .pooling import max_pool_param_checks
from .outputs import set_output_v1_variables
from .padding import check_conv_and_max_pool_compatibility
from .blocks import get_block_out_quantizer
from .block_converter import BlockConverter, register_conversion_patterns
from .conv_common import (parse_conv_block_v1, parse_conv_block_v2,
                          set_conv_variables_v1, set_conv_variables_v2)


__all__ = ["InputConvBlockConverterV1", "InputConvBlockConverterV2"]

_PATTERNS_V1 = [(QuantizedConv2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedReLU)]
_PATTERNS_V2 = [(QuantizedConv2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedReLU),
                (QuantizedConv2D, QuantizedActivation),
                (QuantizedConv2D, QuantizedMaxPool2D, QuantizedActivation)]


def convert_inputconv_block_v1(model_ak, block, inbound_layers_ak):
    """Converts a convolutional block into an akida InputConvolutional layer.

    The expected sequence is:

    - QuantizedConv2D,
    - QuantizedMaxPooling2D (optional),
    - QuantizedReLU.

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """

    conv = block[0]

    # Evaluate the convolutional block layers parameters
    layer_params = parse_conv_block_v1(block, input_layer=True)

    conv_ak = InputConvolutional(**layer_params)

    # Add layer to the model to build its internal variables
    model_ak.add(conv_ak, inbound_layers_ak)
    # Set base variables
    set_conv_variables_v1(conv_ak, conv, input_layer=True)

    # Get out_quantizer of the block.
    out_quantizer = get_block_out_quantizer(block)

    if out_quantizer:
        set_output_v1_variables(conv_ak, out_quantizer)


def convert_inputconv_block_v2(model_ak, block, inbound_layers_ak):
    """Converts a convolutional block into an akida InputConv2D layer.

    The expected sequence is:

    - QuantizedConv2D,
    - QuantizedMaxPooling2D (optional),
    - QuantizedReLU.

    Args:
        model_ak (:obj:`akida.Model`): the target Akida model.
        block (list(:obj:`keras.Layer`)): the block layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Evaluate the convolutional block layers parameters
    layer_params = parse_conv_block_v2(block, input_layer=True)

    conv_ak = InputConv2D(**layer_params)

    # Add layer to the model to build its internal variables
    model_ak.add(conv_ak, inbound_layers_ak)
    # Set base variables
    set_conv_variables_v2(conv_ak, block, flip=False)


class InputConvBlockConverterV1(BlockConverter):
    """Main class that should be used to check if the input conv block is compatible to an Akida v1
    conversion and provides a method to convert it in an equivalent Akida InputConvolutional layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._inputconv_additional_checks()

    def _inputconv_additional_checks(self):
        conv = self._block[0]
        # Get conv layer input shape without the batch size.
        input_shape = conv.input_shape[1:]
        # InputConvolutional handles only image-like inputs (1-D or 3-D uint8 inputs)
        if input_shape[-1] not in (1, 3):
            raise ValueError("InputConvolutional layer can only handle image like inputs (i.e 3-D "
                             f"or 1-D input shape). Received: {input_shape}.")

        # The only weight bitwidth supported is [4, 8]
        weight_bits = conv.weight_quantizer.bitwidth
        if weight_bits not in [4, 8]:
            raise ValueError("InputConvolutional layer can only handle weights with 4 or 8 bits."
                             f" Received: {weight_bits}.")

        next_layer = self._block[1]
        if isinstance(next_layer, QuantizedMaxPool2D):
            check_conv_and_max_pool_compatibility(conv, next_layer)

    def convert(self, model_ak, inbounds):
        convert_inputconv_block_v1(model_ak, self._block, inbounds)


class InputConvBlockConverterV2(BlockConverter):
    """Main class that should be used to check if the input conv block is compatible to an Akida v2
    conversion and provides a method to convert it in an equivalent Akida InputConv2D layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._inputconv_additional_checks()

    def _inputconv_additional_checks(self):
        conv = self._block[0]
        next_layer = self._block[1]
        if isinstance(next_layer, QuantizedMaxPool2D):
            check_conv_and_max_pool_compatibility(conv, next_layer)
            max_pool_param_checks(next_layer)

        # The only weight bitwidth supported is [4, 8]
        weight_bits = conv.weight_quantizer.bitwidth
        if weight_bits not in [4, 8]:
            raise ValueError("InputConvolutional layer can only handle weights with 4 or 8 bits."
                             f" Received: {weight_bits}.")

        # Make sure the Conv2D layers spatial params are square
        assert conv.kernel_size[0] == conv.kernel_size[1], ("InputConv2D handle only"
                                                            "square kernels")
        assert conv.strides[0] == conv.strides[1], ("InputConv2D stride should be"
                                                    "the same for both dimensions")

    def convert(self, model_ak, inbounds):
        convert_inputconv_block_v2(model_ak, self._block, inbounds)


# Register the valid input conv block pattern for Akida v1 and v2
register_conversion_patterns(AkidaVersion.v1, _PATTERNS_V1, InputConvBlockConverterV1, True)
register_conversion_patterns(AkidaVersion.v2, _PATTERNS_V2, InputConvBlockConverterV2, True)
