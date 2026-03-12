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
"""Utility function to convert block layers to akida.
"""
import warnings
from quantizeml.layers import (QuantizedReLU, QuantizedExtractToken, InputQuantizer, Dequantizer,
                               QuantizedPicoPostProcessing)
from ..akida_versions import AkidaVersion, get_akida_version


def split_model_into_blocks(model):
    """Search into the model the sets of possible blocks and return them.

    A set of layers are considered as block if:
        1. There is only one output quantizer (in the last layer of the block)
        2. There are no multi-inbound layers
        3. There are no QuantizedExtractToken layers

    Layers with more than one inbound layer will be handled separately.

    Args:
        model (:obj:`tf.models.Model`): The model to split

    Returns:
        list: a list of sequences of layers ('blocks').
    """
    def _get_inbound_layers(target_layer):
        inbound = target_layer.inbound_nodes[0].inbound_layers
        return inbound if isinstance(inbound, (list, tuple)) else [inbound]

    def _check_single_layer_block(target_layer):
        # Single blocks will be multi-inputs and QuantizedExtractToken (with axis==1)
        prev_layers = _get_inbound_layers(target_layer)
        if len(prev_layers) > 1:
            return True
        if (isinstance(target_layer, QuantizedExtractToken) and target_layer.axis == 1):
            return True
        return False

    def _search_block_v1(target_layer):
        # We consider the target is part of one block if there are previous
        # layers. (Akida v1 doesn't support multi inbound layers)
        prev_layers = _get_inbound_layers(target_layer)
        # If it's the first layer return it
        if len(prev_layers) == 0:
            return [target_layer]

        # If previous layer has an output quantizer or an input quantizer, the blocks end here.
        next_target = prev_layers[0]
        out_quantizer = getattr(next_target, "out_quantizer", None)
        if out_quantizer:
            if not isinstance(next_target, QuantizedReLU):
                raise RuntimeError("Model incompatible with akida v1 version. The layer"
                                   f" {next_target.name} followed by {target_layer.name}"
                                   " is not valid.")
            return [target_layer]
        elif isinstance(next_target, InputQuantizer):
            return [target_layer]

        # Otherwise, we continue searching...
        return _search_block_v1(next_target) + [target_layer]

    def _search_block_v2(target_layer):
        # We consider the target is part of one block if there are previous
        # layers (target is not the model's input), or it's multi-inbound.
        prev_layers = _get_inbound_layers(target_layer)
        if len(prev_layers) != 1:
            return [target_layer]
        # If previous layer has an output quantizer or an input quantizer, the blocks end here.
        next_target = prev_layers[0]
        out_quantizer = getattr(next_target, "out_quantizer", None)
        # Note that the out_quantizer marks the limit of a block only if it has a
        # bitwidth <9
        if (((out_quantizer and out_quantizer.bitwidth < 9)
                or _check_single_layer_block(next_target)) or
                isinstance(next_target, InputQuantizer)):
            return [target_layer]

        # Otherwise, we continue searching...
        return _search_block_v2(next_target) + [target_layer]

    def _search_block(layer):
        return _search_block_v1(layer) if get_akida_version() == AkidaVersion.v1 \
            else _search_block_v2(layer)

    def _get_deq_id(model):
        """Helper to get the first dequantizer id.

        Args:
            model (:obj:`tf.models.Model`): the model to analyse.

        Returns:
            int: the first dequantized id or -1 if not found.
        """
        for layer_id, layer in enumerate(model.layers):
            if isinstance(layer, Dequantizer):
                return layer_id

        return -1

    def _get_layers_before_dequantizer(model):
        # Split layers on the first dequantizer layer
        deq_id = _get_deq_id(model)
        if deq_id >= 0:
            layers_pred = model.layers[:deq_id]
            layers_next = model.layers[deq_id + 1:]
            deq_layer = model.layers[deq_id]
        else:
            layers_pred = model.layers
            layers_next = []
            deq_layer = None

        return deq_layer, layers_pred, layers_next

    def _is_end_of_block(layer, blocks):
        # Return true if the layer is the end of a new block.
        # The very last layer is the end of another block
        if not blocks:
            return True
        outbounds = layer.outbound_nodes
        # If there are many outbounds, this is an end of block
        if len(outbounds) > 1:
            return True
        # Prepare a list of all first layers of all blocks
        first_layers = [block[0] for block in blocks]
        outbound = outbounds[0].outbound_layer
        return outbound in first_layers

    blocks = []
    # If there is a Dequantizer, save the first one as single block layer and then
    # remove all layers after it (not supported at conversion time)
    deq_layer, model_layers, skip_layers = _get_layers_before_dequantizer(model)
    if deq_layer:
        # Drop Dequantizer if preceded by QuantizedPicoPostProcessing
        inbound_layers = _get_inbound_layers(deq_layer)
        if (not (len(inbound_layers) == 1 and
                 isinstance(inbound_layers[0], QuantizedPicoPostProcessing))):
            blocks.append([deq_layer])

    # Prints a warning message with a summary of the skipped layers
    if len(skip_layers) > 0:
        stop_layer_msg = " at layer " + model_layers[-1].name if len(model_layers) > 0 else ""
        skip_layers_summary = "___________________________________________________\n"
        skip_layers_summary += "Layer (type)\n"
        skip_layers_summary += "===================================================\n"
        for layer in skip_layers:
            skip_layers_summary += layer.name + " (" + layer.__class__.__name__ + ")\n"
        skip_layers_summary += "===================================================\n"
        warnings.warn(f"Conversion stops {stop_layer_msg} because of a dequantizer. "
                      f"The end of the model is ignored:\n {skip_layers_summary}. \n This can be "
                      "expected for model heads (e.g. softmax for classification) but could also "
                      "mean that processing layers were not quantized.")

    # We forward from head to bottom
    for layer in model_layers[::-1]:
        if _is_end_of_block(layer, blocks):
            new_block = _search_block(layer)
            blocks.append(new_block)
        # The skip layers must be inside of one block
        elif layer not in sum(blocks, []):
            raise RuntimeError(f"Impossible to append {layer.name} into a block.")

    # At the end, we sort blocks from bottom to head
    return blocks[::-1]


def get_block_out_quantizer(block):
    """Extract the output_quantizer of the layers block.

    Note that a block of layers is considered as block if:
        1. There is only one output quantizer (in the last layer of the block)
        2. There are no multi-inbound layers
        3. There are no QuantizedExtractToken layers

    This method should ideally be called after "split_model_into_blocks" to make sure
    that the layers block can be handled correctly.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block.

    Returns:
        :obj:`quantizeml.layers.OutputQuantizer`: the layers block output quantizer (None if the
            block has no output quantizer)
    """
    # Output quantizer is on the last layer of the block
    return getattr(block[-1], "out_quantizer", None)
