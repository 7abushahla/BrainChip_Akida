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
import numpy as np

import akida

from .base_converter import OnnxConverter
from .padding import get_akida_padding
from .weights import broadcast_and_set_variable, set_weight_variables
from .activation import set_relu_variables, set_lut_variables
from .scale_out import set_output_scale_variables


def _check_if_squared(value, name=""):
    assert hasattr(value, "__iter__")
    if value[0] != value[1]:
        raise ValueError(f"{name} are expected to be square.")


def _parse_additional_parameters(converter, layer_params):
    # Pooling
    if pool_type := getattr(converter, 'pool_type', None):
        layer_params["pool_type"] = pool_type
        if converter.pool_type == akida.PoolType.Max:
            layer_params["pool_size"] = converter.pool_size[0]
            layer_params["pool_stride"] = converter.pool_strides[0]
    # Activation
    layer_params["activation"] = converter.activation


def parse_convolutional_parameters(converter, depthwise=False, transpose=False):
    """Parse a converter into akida parameters

    Args:
        converter (OnnxConverter): the converter to extract parameters.
        depthwise (bool, optional): boolean to declare the layer is depthwise. Defaults to False.
        transpose (bool, optional): boolean to declare the layer is transposed. Defaults to False.

    Returns:
        dict: the parameters
    """
    layer_params = {
        "name": converter.name,
        "output_bits": 8,
        "kernel_size": converter.weights["W"].shape[-1]
    }
    # Add padding and kernel_stride parameter if not a Transpose convolution
    filters = converter.weights["W"].shape[1]
    if not transpose:
        layer_params["padding"] = get_akida_padding(converter)
        layer_params["kernel_stride"] = converter.strides[0]
        # Filters in not transposed layer are in first dimension
        filters = converter.weights["W"].shape[0]
    # Add filters parameters if not a Depthwise convolution
    if not depthwise:
        layer_params["filters"] = filters
    _parse_additional_parameters(converter, layer_params)
    return layer_params


def set_convolutional_variables(converter, ak_layer, flip=True, transpose=False):
    """Transfer converter weights to ak_layer.

    Args:
        converter (OnnxConverter): the converter to extract weights.
        ak_layer (akida.Layer): the target Akida model.
        flip (bool, optional): Boolean to flip W and H dimensions. Defaults to True.
        transpose (bool, optional): Boolean to transpose C and F dimensions. Defaults to False.
    """
    assert isinstance(converter, OnnxConverter)

    # Set padding value (in case of InputConv2D)
    if ak_layer.parameters.layer_type == akida.LayerType.InputConv2D:
        pad_value = converter.weights["x_pad_value"]
        if np.any(pad_value < 0) or np.any(pad_value > 255):
            raise ValueError(f"padding_value must be uint8 values. Received: {pad_value}")
        pad_value = np.array(pad_value, dtype="uint8")
        broadcast_and_set_variable(ak_layer.variables, "padding_value", pad_value)

    # Get kernel and transpose them (FCKxKy -> KxKyCF)
    kernel = converter.weights["W"].transpose((2, 3, 1, 0))
    if flip:
        # Kernel need to be flipped
        kernel = np.flip(kernel, axis=(0, 1))

    # Transpose C and F when required
    if transpose:
        kernel = kernel.transpose((0, 1, 3, 2))

    bias = converter.weights.get("bias", None)
    set_weight_variables(ak_layer, kernel, bias)

    # Activation
    if converter.activation == akida.ActivationType.ReLU:
        set_relu_variables(ak_layer, converter.weights.get("max_value", None))
    elif converter.activation == akida.ActivationType.LUT:
        set_lut_variables(ak_layer,
                          converter.weights["ActScale"],
                          converter.weights["ActShift"],
                          converter.weights["LUT_values"])

    # Scale out
    set_output_scale_variables(ak_layer, converter.weights["Scale"], converter.weights["Shift"])


def check_convolution_compatibility(converter):
    """Check convolution compatibility with Akida.

    Args:
        converter (OnnxConverter): the converter to check.
    """
    kernel_shapes = converter.weights["W"].shape[-2:]
    _check_if_squared(kernel_shapes, name=f"{converter.name} kernels")
    _check_if_squared(converter.strides, name=f"{converter.name} strides")
    if getattr(converter, "pool_type", None) == akida.PoolType.Max:
        _check_if_squared(converter.pool_size, name=f"{converter.name} pool sizes")
        _check_if_squared(converter.pool_strides, name=f"{converter.name} pool strides")
