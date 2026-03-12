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
"""Functions to convert QuantizedSeparableConv2D to Akida.
"""
import numpy as np

from akida import SeparableConvolutional
from quantizeml.layers import (QuantizedSeparableConv2D, QuantizedGlobalAveragePooling2D,
                               QuantizedMaxPool2D, QuantizedReLU, WeightQuantizer,
                               AlignedWeightQuantizer)

from ..akida_versions import AkidaVersion
from .pooling import set_gap_variables_v1, gap_params_checks
from .outputs import set_output_v1_variables
from .padding import check_conv_and_max_pool_compatibility
from .blocks import get_block_out_quantizer
from .block_converter import BlockConverter, register_conversion_patterns
from .conv_common import parse_sepconv_block, get_layer_by_type

__all__ = ["SeparableConvBlockConverter"]

_PATTERNS = [(QuantizedSeparableConv2D,), (QuantizedSeparableConv2D, QuantizedReLU),
             (QuantizedSeparableConv2D, QuantizedGlobalAveragePooling2D, QuantizedReLU),
             (QuantizedSeparableConv2D, QuantizedGlobalAveragePooling2D),
             (QuantizedSeparableConv2D, QuantizedMaxPool2D, QuantizedReLU)]


def _set_sepconv_variables(ak_layer, k_layer):
    """Computes and sets the variables for an Akida SeparableConvolutional layer.

    This function converts the variables of a Keras layer and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        ak_layer (:obj:`akida.Layer`): the targeted akida layer.
        k_layer (:obj:`keras.Layer`): the source quantized layer.
    """
    assert isinstance(k_layer.dw_weight_quantizer, WeightQuantizer)
    assert isinstance(k_layer.pw_weight_quantizer, WeightQuantizer)

    variables_ak = ak_layer.variables

    # Get the QuantizedSeparableConv2D weights
    weights_ak = k_layer.dw_weight_quantizer.qweights.value.fp.values.numpy()
    weights_pw_ak = k_layer.pw_weight_quantizer.qweights.value.fp.values.numpy()
    # We require flip depthwise weights
    weights_ak = np.flip(weights_ak, axis=[0, 1])

    # Get the QuantizedSeparableConv2D bias
    if k_layer.use_bias:
        bias_quantizer = k_layer.bias_quantizer
        assert isinstance(bias_quantizer, AlignedWeightQuantizer)
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        # Store bias into the threshold variable
        variables_ak["threshold"] = -bias

    variables_ak["weights"] = weights_ak.astype(np.int8)
    variables_ak["weights_pw"] = weights_pw_ak.astype(np.int8)


def convert_sepconv_block(model_ak, block, inbound_layers_ak):
    """Converts a separable convolutional block into an akida v1 SeparableConvolutional layer.

    The expected sequence is:

        - QuantizedSeparableConv2D,
        - QuantizedMaxPooling2D or QuantizedGlobalAveragePooling2D (optional),
        - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the model will be added.
        block (list(:obj:`keras.Layer`)): the remaining model layers to convert.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """

    sepconv = block[0]

    # Evaluate the separable convolutional layer parameters
    sepconv_params = parse_sepconv_block(block)

    # Create the Akida SeparableConvolutional layer
    sepconv_ak = SeparableConvolutional(**sepconv_params)

    # Add layer to the model to build its internal variables
    model_ak.add(sepconv_ak, inbound_layers_ak)

    # Set variables
    _set_sepconv_variables(sepconv_ak, sepconv)
    # Get the optional GAP layer
    pool_layer = get_layer_by_type(block, QuantizedGlobalAveragePooling2D)
    if pool_layer:
        set_gap_variables_v1(sepconv_ak, pool_layer)

    # Get out_quantizer of the block.
    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        set_output_v1_variables(sepconv_ak, out_quantizer)


class SeparableConvBlockConverter(BlockConverter):
    """Main class that should be used to check if the sepconv block is compatible to an Akida v1
    conversion and provides a method to convert it in an equivalent Akida SeparableConvolutional
    layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._sepconv_additional_checks()

    def _sepconv_additional_checks(self):
        sep_conv = self._block[0]

        # The only weight bitwidth supported is 4
        weight_bits = sep_conv.dw_weight_quantizer.bitwidth
        if weight_bits not in [1, 2, 4]:
            raise ValueError("SeparableConvolutional layer can only handle 1, 2, or 4 weight bits."
                             f" Received: {weight_bits}.")
        if weight_bits != sep_conv.pw_weight_quantizer.bitwidth:
            raise ValueError("SeparableConvolutional layer pointwise weights should also be 4 bits."
                             f" Received: {sep_conv.pw_weight_quantizer.bitwidth}.")

        pool_layer = get_layer_by_type(self._block, (QuantizedGlobalAveragePooling2D,
                                                     QuantizedMaxPool2D))
        if isinstance(pool_layer, QuantizedMaxPool2D):
            check_conv_and_max_pool_compatibility(sep_conv, pool_layer)
        elif isinstance(pool_layer, QuantizedGlobalAveragePooling2D):
            gap_params_checks(pool_layer)

    def convert(self, model_ak, inbounds):
        convert_sepconv_block(model_ak, self._block, inbounds)


# Register the valid separable conv block pattern for Akida v1
register_conversion_patterns(AkidaVersion.v1, _PATTERNS, SeparableConvBlockConverter)
