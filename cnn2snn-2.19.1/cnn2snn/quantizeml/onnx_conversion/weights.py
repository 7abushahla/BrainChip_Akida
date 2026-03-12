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
__all__ = ["set_weight_variables"]

import numpy as np
from onnx import numpy_helper as np_onnx
from onnxruntime.quantization.quant_utils import find_by_name


def broadcast_and_set_variable(variables, var_name, value):
    """Adapts variables to akida variables shapes if necessary, and sets them.

    Args:
        variable (akida.Variables): the targeted akida variables.
        var_name (str): then name of the variable
        new_value (int or np.ndarray): the new value of the variable
    """
    old_var = variables[var_name]
    # If the variable is a scalar, broadcast it across the target akida variable's last dimension
    new_var = np.full(old_var.shape, value, dtype=old_var.dtype) if np.size(value) == 1 else value
    # Then reshape with expect shape
    variables[var_name] = new_var.reshape(old_var.shape)


def to_value_shift(x, bitwidth=8, signed=True):
    """Transform a numpy array into a 8bit shift approximation.

    Args:
        x (np.ndarray): the array to transform
        bitwidth (int, optional): the maximal value representation. Defaults to 8.
        signed (bool, optional): whether the representation is signed or not. Defaults to True.

    Returns:
        np.ndarray, np.ndarray: the value and shift to represent input as value << shift.
    """
    if signed:
        # Reserve one bit to the sign
        bitwidth -= 1
    # Calculate integer bits, adding an epsilon in order to fix expect closed power of two
    # (e.g. 4 needs 3 integers and not np.ceil(np.log2(4)) == 2)
    # Note: we replace 0 by 2**bitwidth to avoid divide by zero
    y = np.where(x == 0, 2**bitwidth, x)
    int_bits = np.ceil(np.log2(np.abs(y)) + 1e-6)
    # Shift always have to be positive in this representation
    shift = np.maximum(0, int_bits - bitwidth)
    value = np.round(x / 2.0 ** shift)
    # Clip value to representable range
    value = np.clip(value, -2 ** bitwidth if signed else 0, 2 ** bitwidth - 1)
    return value, shift


def set_weight_variables(ak_layer, kernel, bias=None):
    """Set weights into akida variables.

    Args:
        ak_layer (akida.Layer): the akida layer to set variables.
        kernel (np.ndarray): the weights.
        bias (np.ndarray): if applies, the bias. Defaults to None.
    """
    # Set weights without broadcast: reshape/transpose must happen at conversion time
    ak_layer.variables["weights"] = kernel
    if bias is not None:
        # Bias must be converted to a fixed point in the format: signed 8bit + unsigned shift
        ak_bias, ak_bias_shift = to_value_shift(bias)
        ak_bias = np.array(ak_bias, dtype="int8")
        ak_bias_shift = np.array(ak_bias_shift, dtype="uint8")
        broadcast_and_set_variable(ak_layer.variables, "bias", ak_bias)
        broadcast_and_set_variable(ak_layer.variables, "bias_shift", ak_bias_shift)


def load_weights(node, initializers, func=None):
    """Parse model initializers that match with node inputs into numpy arrays.

    Args:

        node (NodeProto): the input node.
        initializers (list of TensorProto): the initializer list.
        func (FunctionProto, optional): the function that represents the node operation.
            If provided, the initializer names will have the inputs of the function
            instead of theirs own ones. Default None.

    Returns:
        dict: the initializers that were found.
    """
    fnames = func.input if func else node.input
    assert len(fnames) == len(node.input), "Mismatch between function and node inputs"

    # Align weight names with function input names
    weights = {}
    for name, fname in zip(node.input, fnames):
        weight = find_by_name(name, initializers)
        if weight:
            weights[fname] = np_onnx.to_array(weight)
    return weights
