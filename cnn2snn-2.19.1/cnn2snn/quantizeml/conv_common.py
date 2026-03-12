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
"""Helper functions to convert QuantizedConv2D to its equivalent Akida layers.
"""
import numpy as np
from akida import PoolType, ActivationType, LayerType
from quantizeml.layers import (QuantizedGlobalAveragePooling2D, QuantizedMaxPool2D,
                               QuantizedReLU, QuantizedActivation,
                               WeightQuantizer, AlignedWeightQuantizer)
from .activations import (parse_relu_v1, parse_relu_v2, set_relu_variables, v1_relu_checks,
                          parse_activation_v2, set_lut_variables)
from .pooling import parse_max_pool_v1, parse_max_pool_v2
from .padding import get_padding, get_padding_value
from .weights import broadcast_and_set_variable
from .blocks import get_block_out_quantizer
from .outputs import set_output_v2_variables, parse_output_bits, parse_post_op_buffer_bits


def get_layer_by_type(block, type_list):
    """Helper to get the first layer, if found, of a block that match one of the types

    Args:
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
        type_list (tuple): list of quantizeml layers types.

    Returns:
        :obj:`keras.Layer`: the first quantized layer that match of the types of the list
        if found. None otherwise.
    """
    targeted_layer = None

    for layer in block:
        if isinstance(layer, type_list):
            targeted_layer = layer
            break

    return targeted_layer


def set_conv_variables_v1(layer_ak, layer_k, input_layer=False):
    """Computes and sets the variables for an Akida v1 Convolutional layer.

    This function converts the variables of a Keras layer and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        layer_k (:obj:`keras.Layer`): the source quantized layer.
        input_layer (bool, optional): Boolean to check if the targeted layer is a
            V1 conv input layer (i.e InputConvolutional). Defaults to False.

    """
    assert isinstance(layer_k.weight_quantizer, WeightQuantizer)
    if layer_k.use_bias:
        assert isinstance(layer_k.bias_quantizer, AlignedWeightQuantizer)

    # Get the weights
    weights = layer_k.weight_quantizer.qweights.value.fp.values.numpy()
    # Flip W and H dimensions for conv. kernels (not input conv.)
    if not input_layer:
        weights = np.flip(weights, axis=[0, 1])
    layer_ak.variables["weights"] = weights.astype(np.int8)

    # Set the bias (if there is one)
    if layer_k.use_bias:
        bias_quantizer = layer_k.bias_quantizer
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        # Store bias into the threshold variable
        layer_ak.variables["threshold"] = -bias


def set_conv_variables_v2(layer_ak, block, flip=True):
    """Computes and sets the variables for an Akida v2 Convolutional layer.

    This function converts the variables of a Keras layer and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        block (list(:obj:`keras.Layer`)): the (depthwise)conv2d(_transpose) block layers.
        flip (bool, optional): Boolean to flip W and H dimensions. Defaults to True.
    """
    layer_k = block[0]

    assert isinstance(layer_k.weight_quantizer, WeightQuantizer)
    if layer_k.use_bias:
        assert isinstance(layer_k.bias_quantizer, AlignedWeightQuantizer)

    # Get the weights
    weights = layer_k.weight_quantizer.qweights.value.fp.values.numpy()
    # Flip W and H dimensions for conv. kernels (not input/transposed conv)
    if flip:
        weights = np.flip(weights, axis=[0, 1])
    layer_ak.variables["weights"] = weights.astype(np.int8)

    # Set the bias (if there is one)
    if layer_k.use_bias:
        bias_quantizer = layer_k.bias_quantizer
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        bias_shift = bias_quantizer.shift.value.numpy().astype(np.uint8)
        # Unshift the bias and store it
        layer_ak.variables["bias"] = (bias >> bias_shift).astype(np.int8)
        # Also store the bias shift
        broadcast_and_set_variable(layer_ak.variables, "bias_shift", bias_shift)

    # Set input shift
    input_shift = getattr(layer_k, 'input_shift', None)
    if input_shift is not None:
        input_shift = input_shift.value.numpy().astype(np.uint8)
        if layer_ak.parameters.layer_type != LayerType.InputConv2D:
            broadcast_and_set_variable(layer_ak.variables, "input_shift", input_shift)
        elif np.any(input_shift != 0):
            raise ValueError("InputConv2D layer must only have non-zero input shift.")

    # Set padding value (in case of InputConv2D)
    padding_value = get_padding_value(layer_k).astype("uint8")
    if layer_ak.parameters.layer_type == LayerType.InputConv2D:
        broadcast_and_set_variable(layer_ak.variables, "padding_value", padding_value)
    elif np.any(padding_value != 0):
        # For non-input conv, padding value must be equal to zero.
        raise ValueError(f"Conv2D padding value must be zero. Got {padding_value}.")

    # Check if we have activation
    activation_layer = get_layer_by_type(block, (QuantizedReLU, QuantizedActivation))
    if isinstance(activation_layer, QuantizedReLU):
        set_relu_variables(layer_ak, activation_layer)
    if isinstance(activation_layer, QuantizedActivation):
        set_lut_variables(layer_ak, layer_k, activation_layer)

    # Get out_quantizer of the block.
    out_quantizer = get_block_out_quantizer(block)

    if out_quantizer:
        set_output_v2_variables(layer_ak, out_quantizer)


def _parse_additional_layers_v1(block, block_params):
    """Parse conv block additional layers into the parameters of one for Akida v1.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block.
        block_params (dict): the current block parameters.

    """
    # Identify the next layers

    relu_layer = get_layer_by_type(block, QuantizedReLU)
    pool_layer = get_layer_by_type(block, (QuantizedGlobalAveragePooling2D, QuantizedMaxPool2D))

    if isinstance(pool_layer, QuantizedMaxPool2D):
        pool_params = parse_max_pool_v1(pool_layer)
        block_params.update(pool_params)
    elif isinstance(pool_layer, QuantizedGlobalAveragePooling2D):
        pool_params = dict(pool_type=PoolType.Average)
        block_params.update(pool_params)
    if relu_layer:
        v1_relu_checks(relu_layer)
        act_params = parse_relu_v1(relu_layer)
        block_params.update(act_params)


def _parse_additional_layers_v2(block, block_params):
    """Parse conv block additional layers into the parameters of one for Akida v2.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block.
        block_params (dict): the current block parameters.

    """
    # Identify the next layers

    activation_layer = get_layer_by_type(block, (QuantizedReLU, QuantizedActivation))
    pool_layer = get_layer_by_type(block, (QuantizedGlobalAveragePooling2D, QuantizedMaxPool2D))

    if isinstance(pool_layer, QuantizedMaxPool2D):
        pool_params = parse_max_pool_v2(pool_layer)
        block_params.update(pool_params)
    elif isinstance(pool_layer, QuantizedGlobalAveragePooling2D):
        block_params.update({"pool_type": PoolType.Average})
    if isinstance(activation_layer, QuantizedReLU):
        act_params = parse_relu_v2(activation_layer)
        block_params.update(act_params)
    elif isinstance(activation_layer, QuantizedActivation):
        act_params = parse_activation_v2(activation_layer)
        block_params.update(act_params)
    # parse the block output bits
    parse_output_bits(block, block_params)
    # parse the block post op buffer bits
    parse_post_op_buffer_bits(block, block_params)


def parse_conv_block_v1(block, input_layer=False):
    """Parses a conv block parameters for an Akida v1 layer.

    Args:
        block (list(:obj:`keras.Layer`)): the conv block layers.
        input_layer (bool, optional): Boolean to check if the targeted layer is a
            V1 conv input layer (i.e InputConvolutional). Defaults to False.

    Returns:
        dict: the corresponding akida parameters.
    """
    conv = block[0]
    conv_params = dict(
        padding=get_padding(conv),
        kernel_size=(conv.kernel_size[0], conv.kernel_size[1]),
        filters=int(conv.kernel.shape[3]),
        weights_bits=conv.weight_quantizer.bitwidth,
        kernel_stride=(conv.strides[0], conv.strides[1]),
        activation=False,
        name=conv.name
    )
    # If the targeted layer is a InputConvolutional add two params.
    p_value = get_padding_value(conv)
    if input_layer:
        if np.size(p_value) > 1 and np.all(p_value[0] == p_value):
            p_value = p_value[0]
        if np.size(p_value := np.squeeze(p_value)) > 1:
            raise ValueError("InputConvolutional cannot have per-axis padding value. "
                             f"Got {p_value}.")
        input_layer_params = dict(input_shape=tuple(conv.input_shape[1:4]), padding_value=p_value)
        conv_params.update(input_layer_params)
    elif np.any(p_value != 0):
        # For non-input conv, padding value must be equal to zero.
        raise ValueError(f"Convolutional padding value must be zero. Got {p_value}.")

    _parse_additional_layers_v1(block, conv_params)
    return conv_params


def parse_conv_block_v2(block, input_layer=False, depthwise=False, transpose=False):
    """Parses a conv block parameters for akida v2.

    Args:
        block (list(:obj:`keras.Layer`)): the conv block layers.
        input_layer (bool, optional): Boolean to check if the targeted layer is a
            V2 conv input layer (i.e InputConv2D). Defaults to False.
        depthwise (bool, optional): boolean to declare if layer is depthwise. Defaults to False.
        transpose (bool, optional): boolean to declare if layer is transposed. Defaults to False.

    Returns:
        dict: the corresponding akida parameters.
    """
    conv = block[0]

    # In quantizeml one bit is reserved for the sign in the buffer bitwidth
    # variable, but in akida this value has to be added back to have the
    # correct clipping.
    conv_params = dict(
        kernel_size=conv.kernel_size[0],
        activation=ActivationType.NoActivation,
        buffer_bits=conv.buffer_bitwidth + 1,
        weights_bits=conv.weight_quantizer.bitwidth,
        name=conv.name
    )
    # If the targeted layer is a InputConv2D add two params.
    if input_layer:
        conv_params["input_shape"] = tuple(conv.input_shape[1:4])
    # Add filters parameters if not a Depthwise convolution
    if not depthwise:
        conv_params["filters"] = conv.filters
    # Add padding and kernel_stride parameter if not a Transpose convolution
    if not transpose:
        conv_params["padding"] = get_padding(conv)
        conv_params["kernel_stride"] = conv.strides[0]

    _parse_additional_layers_v2(block, conv_params)
    return conv_params


def parse_sepconv_block(block):
    """Parses a quantizeml sepconv block parameters for Akida v1.

    Args:
        block (list(:obj:`keras.Layer`)): the sepconv block layers.

    Returns:
        dict: the corresponding akida parameters.
    """
    sepconv = block[0]
    # Padding value must be built in constructor
    padding = get_padding(sepconv)

    sepconv_params = dict(kernel_size=(sepconv.kernel_size[0], sepconv.kernel_size[1]),
                          filters=int(sepconv.pointwise_kernel.shape[3]),
                          padding=padding,
                          kernel_stride=(sepconv.strides[0], sepconv.strides[1]),
                          weights_bits=sepconv.dw_weight_quantizer.bitwidth,
                          activation=False,
                          name=sepconv.name)

    _parse_additional_layers_v1(block, sepconv_params)
    return sepconv_params
