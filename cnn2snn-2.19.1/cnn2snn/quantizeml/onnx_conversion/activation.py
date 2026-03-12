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
__all__ = ["set_relu_variables", "set_lut_variables", "parse_activation_type"]

import akida
import numpy as np

from .weights import broadcast_and_set_variable, to_value_shift
from .scale_out import set_output_scale_variables


def set_relu_variables(ak_layer, max_value=None):
    """Set max value into akida variables.

    Args:
        ak_layer (akida.Layer): the akida layer to set variables.
        max_value (int, optional): the maximal value to set. Defaults to None.
    """
    if max_value is not None:
        # Max value is converted to a fixed point: unsigned 8bit << unsigned 8bit
        max_value, max_value_shift = to_value_shift(max_value, signed=False)
        max_value = np.array(max_value, dtype="uint8")
        max_value_shift = np.array(max_value_shift, dtype="uint8")
        broadcast_and_set_variable(ak_layer.variables, "max_value", max_value)
        broadcast_and_set_variable(ak_layer.variables, "max_value_shift", max_value_shift)


def set_lut_variables(ak_layer, act_scale, act_shift, lut_values):
    """Set activation variables into akida layer.

    Args:
        ak_layer (akida.Layer): the akida layer to set variables.
        act_scale (np.ndarray): activation scale.
        act_shift (np.ndarray): activation shift.
        lut_values (np.ndarray): activation table.
    """
    set_output_scale_variables(ak_layer, act_scale, act_shift, prefix="activation")
    # LUT_values in ONNX follow the indices format : [0, 1, ... N-1, -N, -(N-1), ..., -1]
    # But akida expects : [-N, -(N-1), ..., -1, 0, 1, ... N-1]
    N = lut_values.shape[0] // 2
    lut_values = np.concatenate([lut_values[N:], lut_values[:N]])
    broadcast_and_set_variable(ak_layer.variables, "lut_values", lut_values)


def parse_activation_type(op_type):
    """Parse activation type from op_type to akida.ActivationType.

    Args:
        op_type (str): the operator type.

    Returns:
        ActivationType: the corresponding Akida activation type.
    """
    akida_act_type = akida.ActivationType.NoActivation
    if "ReLU" in op_type:
        akida_act_type = akida.ActivationType.ReLU
    elif "LUT" in op_type:
        akida_act_type = akida.ActivationType.LUT
    return akida_act_type
