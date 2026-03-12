#!/usr/bin/env python
# ******************************************************************************
# Copyright 2022 Brainchip Holdings Ltd.
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
"""Functions to set weights from quantizeml quantized Keras models to Akida.
"""
import numpy as np


def broadcast_and_set_variable(variables, var_name, value):
    """Adapts variables to akida variables shapes if necessary, and sets them.

    Args:
        variable (:obj:`akida.Variables`): the targeted akida variables.
        var_name (str): name of the variable
        value (int): value of the variable
    """
    var = variables[var_name]
    # If the variable is a scalar, broadcast it across the target akida
    # variable's last dimension.
    shape = var.shape
    if np.size(value) == 1:
        variables[var_name] = np.full(shape, value)
    else:
        variables[var_name] = np.reshape(value, shape)


def broadcast_and_set_variables(layer_ak, variables):
    """Adapts variables to akida variables shapes if necessary, and sets them.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        variables (dict): dictionary of variables
    """
    for var_name in variables:
        value = variables[var_name]
        broadcast_and_set_variable(
            layer_ak.variables, var_name, value)
