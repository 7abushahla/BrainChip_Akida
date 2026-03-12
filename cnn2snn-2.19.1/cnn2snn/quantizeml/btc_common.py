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
"""Helper functions to convert (Depthwise)BufferTempConv layers to their equivalent Akida layers.
"""
import numpy as np
from akida import ActivationType
from quantizeml.layers import (QuantizedReLU, WeightQuantizer,
                               AlignedWeightQuantizer, QuantizedBufferTempConv)
from .conv_common import get_layer_by_type
from .activations import parse_relu_v2, set_relu_variables
from .weights import broadcast_and_set_variable
from .blocks import get_block_out_quantizer
from .outputs import set_output_v2_variables, parse_output_bits, parse_post_op_buffer_bits


def parse_buf_temp_conv_block(block, depthwise=False):
    """Parses a quantizeml (depthwise) buffer_temp_conv block parameters for Akida v2.

    Args:
        block (list(:obj:`keras.Layer`)): the buffer_temp_conv block layers.
        depthwise (bool, optional): boolean to declare the main layer as a depthwise layer.
            Defaults to False.

    Returns:
        dict: the corresponding Akida parameters.
    """

    btc_layer = block[0]

    # In quantizeml one bit is reserved for the sign in the buffer bitwidth
    # variable, but in akida this value has to be added back to have the
    # correct clipping.
    buffer_bits = btc_layer.buffer_bitwidth + 1

    block_params = dict(
        fifo_size=btc_layer.kernel_size,
        buffer_bits=buffer_bits,
        activation=ActivationType.NoActivation,
        weights_bits=btc_layer.weight_quantizer.bitwidth,
        name=btc_layer.name
    )
    # Add filters parameters if not a Depthwise BTC
    if not depthwise:
        assert isinstance(btc_layer, QuantizedBufferTempConv)
        block_params["filters"] = btc_layer.filters

    relu_layer = get_layer_by_type(block, QuantizedReLU)
    if relu_layer:
        act_params = parse_relu_v2(relu_layer)
        block_params.update(act_params)

    # parse the block output bits
    parse_output_bits(block, block_params)
    # parse the block post op buffer bits
    parse_post_op_buffer_bits(block, block_params)

    return block_params


def set_buffer_temp_conv_block_variables(layer_ak, block, depthwise=False):
    """Computes and sets the variables for an Akida v2 BufferTempConv layers.

    This function converts the variables of a Keras layers block and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        block (list(:obj:`keras.Layer`)): the buffer temporal convolution block layers.
        depthwise (bool, optional): boolean to declare the main layer as a depthwise layer.
            Defaults to False.
    """
    btc_layer = block[0]

    assert isinstance(btc_layer.weight_quantizer, WeightQuantizer)
    if btc_layer.use_bias:
        assert isinstance(btc_layer.bias_quantizer, AlignedWeightQuantizer)

    # Get the weights
    weights = btc_layer.weight_quantizer.qweights.value.fp.values.numpy()
    if depthwise:
        assert len(weights.shape) == 2, "expected depthwise layer weights with a shape (T, C)"
        # Expand weights shape to match Akida expectation
        weights = weights.reshape((1, 1, *weights.shape))
    layer_ak.variables["weights"] = weights.astype(np.int8)

    # Set the bias (if there is one)
    if btc_layer.use_bias:
        bias_quantizer = btc_layer.bias_quantizer
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        bias_shift = bias_quantizer.shift.value.numpy().astype(np.uint8)

        # Unshift the bias and store it
        layer_ak.variables["bias"] = (bias >> bias_shift).astype(np.int8)
        # Also store the bias shift
        broadcast_and_set_variable(layer_ak.variables, "bias_shift", bias_shift)

    # Set input shift if available
    if getattr(btc_layer, 'input_shift', None):
        broadcast_and_set_variable(layer_ak.variables, "input_shift",
                                   btc_layer.input_shift.value.numpy().astype(np.uint8))

    # Check if we have ReLU
    relu_layer = get_layer_by_type(block, QuantizedReLU)
    # Set optional activation variables
    if relu_layer:
        set_relu_variables(layer_ak, relu_layer)

    # Get the layer block output_quantizer
    out_quantizer = get_block_out_quantizer(block)
    # Set optional output_quantizer variables
    if out_quantizer:
        set_output_v2_variables(layer_ak, out_quantizer)
