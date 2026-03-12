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
"""Functions to update akida layer output variables from a keras OutputQuantizer.
"""
import numpy as np
import tensorflow as tf
from quantizeml.layers import OutputQuantizer
from quantizeml.tensors import pow2
from .weights import broadcast_and_set_variable
from .blocks import get_block_out_quantizer


def parse_output_bits(block, block_params):
    """ Helper to set the block output bits in its corresponding dict block_params.
    Note that this helper should be called only once the buffer_bits is set. What is normally done
    when the main block layer is parsed.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block.
        block_params (dict): the current block parameters.
    """
    # Get the layer block output_quantizer
    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        block_params.update({"output_bits": out_quantizer.bitwidth})
    else:
        # Default to buffer bitwidth
        block_params.update({'output_bits': block_params["buffer_bits"]})


def parse_post_op_buffer_bits(block, block_params):
    """ Helper to set the block post op buffer bits in its corresponding dict block_params.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block.
        block_params (dict): the current block parameters.
    """
    # Get the layer block output_quantizer
    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        block_params.update({"post_op_buffer_bits": out_quantizer.buffer_bitwidth})


def set_output_v1_variables(layer_ak, out_quantizer):
    """Computes and sets the output variables in an akida v1 layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida v1 layer.
        out_quantizer (:obj:`quantizeml.OutputQuantizer`): the source output quantizer.
    """
    assert isinstance(out_quantizer, OutputQuantizer)

    # Extract the OutputQuantizer variables
    scales = out_quantizer.qscales.value.values
    shift = out_quantizer.shift.value

    # Quantizeml evaluates the outputs as: y = x * scales * 2^shift
    # Calculate the float activation step, as the reciprocal of (scales * 2^shift)
    scales_rec = (pow2(-shift) / scales).numpy().astype(np.float32)
    # In akida 1.0, the outputs are evaluated as: y = x / act_step
    layer_ak.variables["act_step"] = scales_rec
    if layer_ak.parameters.activation:
        # For activations, x is decreased by half the activations step before division
        # to increase the precision. This is obtained by increasing the threshold.
        layer_ak.variables["threshold"] += np.round(0.5 * scales_rec).astype(np.int32)
        # Adjust activation step to match activation formula
        act_bits = layer_ak.parameters.act_bits
        layer_ak.variables["act_step"] *= 2 ** (act_bits - 4)


def set_output_v2_variables(layer_ak, out_quantizer, prefix="output"):
    """Computes and sets the output variables in an akida v2 layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida v2 layer.
        out_quantizer (:obj:`quantizeml.OutputQuantizer`): the source output quantizer.
        prefix (str, optional): prefix to use for the variable names. Defaults to "output".
    """
    assert isinstance(out_quantizer, OutputQuantizer)

    # Extract the OutputQuantizer variables.
    # Note that qscales exist only if the out_quantizer was used on QFloat (not FixedPoint as in
    # the QuantizedAdd layer)
    if hasattr(out_quantizer, 'qscales'):
        scales = out_quantizer.qscales.value.values
        output_scales = scales.numpy().astype(layer_ak.variables[f"{prefix}_scales"].dtype)
        broadcast_and_set_variable(layer_ak.variables, f"{prefix}_scales", output_scales)

    # clip shift value to ensure that it is within a valid range
    shift = tf.maximum(out_quantizer.shift.value, -layer_ak.parameters.buffer_bits + 1)
    broadcast_and_set_variable(layer_ak.variables, f"{prefix}_shift", shift.numpy().astype(np.int8))
