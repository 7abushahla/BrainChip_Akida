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
"""Functions to convert keras activation layers parameters and variables to akida.
"""
import numpy as np
from akida import ActivationType

from quantizeml.layers import QuantizedReLU, AlignedWeightQuantizer, OutputQuantizer
from .weights import broadcast_and_set_variable
from .outputs import set_output_v2_variables, get_block_out_quantizer


def v1_relu_checks(layer):
    """Additional checks on QuantizedReLU layers of akida v1 models.

    Args:
        layer (:obj:`keras.Layer`): the QuantizedReLU layer to check.
    """
    lname = layer.name
    if layer.max_value is None:
        raise ValueError(f"{lname}: in AkidaVersion.v1, unbounded QuantizedReLU is not supported.")
    assert isinstance(layer.max_value_quantizer, AlignedWeightQuantizer)
    # Check if there is an output_quantizer
    out_quantizer = getattr(layer, "out_quantizer", None)
    if not out_quantizer:
        raise ValueError(f"{lname}: in AkidaVersion.v1, output_quantizer is mandatory.")
    assert isinstance(out_quantizer, OutputQuantizer)
    if out_quantizer._axis == "per-axis":
        raise ValueError(f"{lname}: in AkidaVersion.v1, output_quantizer must be per-tensor.")


def parse_relu_v1(layer_k):
    """Parses the quantizeml.QuantizedReLU parameters for Akida v1 layers.

    Args:
        layer_k (:obj:`keras.Layer`): the QuantizedReLU layer to parse.

    Returns:
        dict: the corresponding akida parameters.
    """
    # Check if there is an output_quantizer
    out_quantizer = getattr(layer_k, "out_quantizer", None)

    # In AkidaVersion.v1, output_quantizer is mandatory
    assert out_quantizer is not None

    return {'activation': True, 'act_bits': out_quantizer.bitwidth}


def parse_relu_v2(layer_k):
    """Parses the quantizeml.QuantizedReLU parameters for Akida v2 layers.

    Args:
        layer_k (:obj:`keras.Layer`): the QuantizedReLU layer to parse.

    Returns:
        dict: the corresponding akida parameters.
    """

    return {'activation': ActivationType.ReLU}


def parse_activation_v2(layer_k):
    """Parses the quantizeml.QuantizedActivation parameters for Akida v2 layers.

    Args:
        layer_k (:obj:`keras.Layer`): the QuantizedActivation layer to parse.

    Returns:
        dict: the corresponding akida parameters.
    """
    return {'activation': ActivationType.LUT}


def set_relu_variables(layer_ak, layer_k):
    """Computes and sets the activation variables in an akida v2 layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        layer_k (:obj:`quantizeml.QuantizedRelu`): the source QuantizedReLU layer.
    """

    assert isinstance(layer_k, QuantizedReLU)

    variables_ak = layer_ak.variables

    if layer_k.max_value is not None:
        max_value_quantizer = layer_k.max_value_quantizer
        assert isinstance(max_value_quantizer, AlignedWeightQuantizer)
        max_value = max_value_quantizer.qweights.value.values.numpy().astype(np.int32)
        max_value_shift = max_value_quantizer.shift.value.numpy().astype(np.uint8)
        max_value_ak = (max_value >> max_value_shift).astype(np.uint8)
        broadcast_and_set_variable(variables_ak, "max_value", max_value_ak)
        broadcast_and_set_variable(variables_ak, "max_value_shift", max_value_shift)


def set_lut_variables(layer_ak, main_layer, activation_layer):
    """Computes and sets the activation variables in an akida v2 layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        main_layer (:obj:`quantizeml.layers.QuantizedLayer`): the main source layer.
        activation_layer (:obj:`quantizeml.QuantizedActivation`): the source
            QuantizedActivation layer.
    """
    # An akida layer that implements a LUT may have:
    # * intermediate scales/shifts
    if out_quantizer := get_block_out_quantizer([main_layer]):
        set_output_v2_variables(layer_ak, out_quantizer, prefix="activation")
    # * lut values
    x, table = activation_layer.values.export()
    # Akida expects a table organized by input indices.
    indices = np.argsort(x)
    table = table.numpy()[indices].astype("int32")
    try:
        broadcast_and_set_variable(layer_ak.variables, "lut_values", table)
    except ValueError:
        # Table size is hardcoded, so the only way to control its size is through quantization.
        raise ValueError(f"output_quantizer on {activation_layer.name} must have a bitwidth of "
                         "11 bits.")
