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
"""Functions to convert QuantizedAdd to Akida.
"""
from akida import Add
import quantizeml.layers as qlayers
import numpy as np

from ..akida_versions import AkidaVersion
from .weights import broadcast_and_set_variable
from .block_converter import BlockConverter, register_conversion_patterns
from .blocks import get_block_out_quantizer
from .outputs import set_output_v2_variables, parse_output_bits, parse_post_op_buffer_bits


__all__ = ["AddBlockConverter"]

_PATTERNS = [(qlayers.QuantizedAdd,)]


def _set_add_variables(ak_layer, block):
    """Computes and sets the variables for an Akida Add layer.

    This function converts the variables of a Keras layer and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        ak_layer (:obj:`akida.Layer`): the targeted akida layer.
        block (list(:obj:`keras.Layer`)): the block of keras layers.
    """
    add_layer = block[0]

    variables_ak = ak_layer.variables
    a_shift = add_layer.a_shift.value.numpy().astype(np.uint8)
    broadcast_and_set_variable(variables_ak, "a_shift", a_shift)
    b_shift = add_layer.b_shift.value.numpy().astype(np.uint8)
    broadcast_and_set_variable(variables_ak, "b_shift", b_shift)

    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        set_output_v2_variables(ak_layer, out_quantizer)


def _create_add(block):
    """Parses a quantizeml QuantizedAdd layer and returns the corresponding
    Akida v2 Add layer.

    Args:
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.

    Returns:
        :obj:`akida.Add`: The created akida layer.
    """
    add_layer = block[0]
    # In quantizeml one reserves automatically one bit for the sign, but in akida
    # this is rather checked during the clipping operations.
    add_params = {"buffer_bits": add_layer.buffer_bitwidth + 1}
    # parse the block output bits
    parse_output_bits(block, add_params)
    # parse the block post op buffer bits
    parse_post_op_buffer_bits(block, add_params)

    return Add(**add_params,
               name=add_layer.name)


def convert_quantized_add_block(model_ak, block, inbound_layers_ak):
    """Converts QuantizedAdd layer block and its variables and adds it to the
    Akida's v2 model.

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the model will be added.
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Create and add layer to the akida model
    layer_ak = _create_add(block)
    model_ak.add(layer_ak, inbound_layers_ak)
    # Set the akida layer converted variables
    _set_add_variables(layer_ak, block)


class AddBlockConverter(BlockConverter):
    """Main class that should be used to check if the Add layer block is compatible to an
    Akida v2 conversion and provides a method to convert it in an equivalent Akida
    Add layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def convert(self, model_ak, inbounds):
        convert_quantized_add_block(model_ak, self._block, inbounds)


# Register the valid add block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, AddBlockConverter)
