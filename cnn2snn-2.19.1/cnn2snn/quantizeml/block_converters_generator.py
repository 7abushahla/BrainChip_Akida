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
"""Functions to generate block converters from a QuantizeML model.
"""
import os
import tensorflow as tf
from tf_keras import layers, Sequential
from quantizeml import layers as qlayers
from quantizeml.models.transforms.transforms_utils import get_layers_by_type

from .blocks import split_model_into_blocks
from ..transforms.sequential import (_check_layers_data_format, _check_layers_dilation,
                                     _check_layer_inbounds)
from ..akida_versions import get_akida_version, AkidaVersion
from .block_converter import (_V1_PATTERN_CONVERTERS, _V2_PATTERN_CONVERTERS,
                              _V1_INPUT_PATTERN_CONVERTERS, _V2_INPUT_PATTERN_CONVERTERS)
from .input_data import get_tensor_dtype
from .input_quantizer import InputQuantizerBlockConverter

skippable_layers = (layers.InputLayer, layers.Rescaling, layers.Activation, layers.Softmax,
                    layers.Dropout)


def _block_pattern(block):
    """Method that returns the pattern of a block of layers.

    Args:
        block (list): list of quantized quantizeml layers.

    Returns:
        tuple: list of layer types representing the block pattern.
    """
    return tuple([layer.__class__ for layer in block])


def _get_block_converter(block):
    """Helper to get the BlockConverter of a block of layers.

    Args:
        block (list): list of quantized quantizeml layers.

    Returns:
        (:obj:`BlockConverter`): the BlockConverter corresponding to the block of layers or None.
    """
    pattern = _block_pattern(block)

    if get_akida_version() == AkidaVersion.v1:
        block_converter = _V1_PATTERN_CONVERTERS.get(pattern, None)
    else:
        block_converter = _V2_PATTERN_CONVERTERS.get(pattern, None)

    return block_converter


def _get_input_block_converter(block, model, straight_blocks):
    """Helper to get the BlockConverter of an input block of layers.

    For a block to be considered an input block:
    - it represents an InputQuantizerBlockConverter or
    - it is the first mappable block of the model (i.e. straight_blocks is empty) and
    - the model input must fit with HRC conditions and
    - quantizer produces unsigned 8-bit output (if there is one).

    Args:
        block (list): list of quantized quantizeml layers.
        model (keras.Model): the model to convert.
        straight_blocks (list): list of already converted blocks.

    Returns:
        (:obj:`BlockConverter`): the BlockConverter corresponding to the block of layers or None.
    """
    # Remove InputQuantizerBlockConverter from straight_blocks since it should not be taken
    # into account in the following conditions.
    if (len(straight_blocks) > 0 and
            isinstance(qconvert := straight_blocks[0], InputQuantizerBlockConverter)):
        # Quantizer must produce uint8 outputs.
        if qconvert.root.signed or qconvert.root.bitwidth != 8:
            return None

        # Remove the quantizer from the straight blocks.
        straight_blocks = straight_blocks[1:]

    # The block must be the first one to be converted.
    if len(straight_blocks) > 0:
        return None

    # Partially convert the block.
    pattern = _block_pattern(block)
    if get_akida_version() == AkidaVersion.v1:
        converter = _V1_INPUT_PATTERN_CONVERTERS.get(pattern, None)
    else:
        converter = _V2_INPUT_PATTERN_CONVERTERS.get(pattern, None)

    # The block must fit with HRC conditions if it is not an InputQuantizerBlockConverter.
    input_dtype = get_tensor_dtype(model.input, signed=False)
    if not ((model.input_shape[-1] in (1, 3) and input_dtype == tf.uint8) or
            converter == InputQuantizerBlockConverter):
        return None

    return converter


def _display_supported_patterns(layer_filter=None):
    """Helper to display the list of the supported patterns

    Args:
        layer_filter (list, optional): a layer list to limit the pattern list. Defaults to None.

    Returns:
        str: the list of supported patterns.
    """
    if get_akida_version() == AkidaVersion.v1:
        pattern_converters = _V1_PATTERN_CONVERTERS
    else:
        pattern_converters = _V2_PATTERN_CONVERTERS
    keys = list(pattern_converters.keys())
    display = ""
    for key in keys:
        layer_pattern = [layer.__name__ for layer in key]
        # Add the pattern if no filter or if there is a common layer
        if layer_filter is None or set(layer_pattern) & set(layer_filter):
            display += " - " + " > ".join(layer_pattern) + "\n"

    # if layer_filter filtered all patterns, display without filter
    if display == "" and len(keys) > 0:
        display = _display_supported_patterns()

    return display


def generate_block_converters(model):
    """Extract block converts from a QuantizeML model

    Exceptions are raised if incompatibilities encountered.

    Args:
        model (:obj:`keras.Model`): the model to extract.

    Returns:
        list: a list of sequences of the non_skippable layers ('blocks').
    """
    # Check general rules about model in three steps:
    # 1. Check if model has only one input and one output,
    # 2. Check right data format and
    # 3. Over Akida 1.0, check if model is sequential.
    _check_model_input_output(model)
    _check_layers_data_format(model)
    _check_layers_dilation(model)
    if get_akida_version() == AkidaVersion.v1:
        _check_layer_inbounds(model)

    # Split model into theirs blocks:
    blocks = split_model_into_blocks(model)

    # This list will contains either a block converter instance,
    # or a list of non-skippable layers.
    straight_blocks = []
    # Evaluate block-by-block integrity
    for block in blocks:
        # Initialize block_converter to None
        block_converter = None
        # Split blocks into skippable and none skippable blocks
        _, non_skippable = _extract_skippable_layers(block)
        # Skip the block if the block contains only skippable layers
        if len(non_skippable) == 0:
            continue
        # Get the corresponding BlockConverter of the layers block if available.
        block_converter = _get_input_block_converter(non_skippable, model, straight_blocks)
        # If the first block doesn't match any Input pattern follow the classical way
        if not block_converter:
            block_converter = _get_block_converter(non_skippable)
        # One shouldn't get in here. If so the block pattern is unconvertible. Raise an error.
        if not block_converter:
            layers_type_names = [type(layer).__name__ for layer in non_skippable]
            supported_patterns = _display_supported_patterns(layers_type_names)
            type_and_name = [f"{type(layer).__name__} ({layer.name})" for layer in non_skippable]
            raise RuntimeError("Invalid block found during conversion:\n"
                               f" {os.linesep.join(type_and_name)}"
                               "\nCompatible patterns that contain the faulty layers are:"
                               f"\n{supported_patterns}")
        straight_blocks.append(block_converter(non_skippable))

    return straight_blocks


def _check_model_input_output(model):
    """Asserts that model inputs/outputs are supported for conversion.

    The Keras model must have only one input layer and one output layer.
    On Akida 1.0, the input shape must 4-D (N, H, W, C).

    Args:
        model (keras.model): the Keras model to check.
    """
    # Error if all model layers are skippable
    if all(isinstance(layer, (*skippable_layers, qlayers.Dequantizer)) for layer in model.layers):
        raise RuntimeError("Cannot convert model with only skippable layers.")

    # If the quantized keras model is Sequential, it should have its input_shape defined.
    if isinstance(model, Sequential):
        input_shape = getattr(model.layers[0], "input_shape", None)
        if input_shape is None:
            raise AttributeError(f"The Sequential quantized model {model.name} should have its"
                                 " input_shape fully defined.")

    # Error if multiple inputs
    if len(model.input_names) > 1:
        raise RuntimeError("Model must have only one input layer. Receives "
                           f"inputs {model.input_names}.")

    # Error if multiple outputs
    if len(model.output_names) != 1:
        raise RuntimeError("Model must have only one output layer. Receives"
                           f"outputs {model.output_names}.")

    # Error if input shape is not 2D or 4D when no StatefulRecurrent layer is present (in this case,
    # inputs are (B, T, C))
    if (len(model.input_shape) not in (2, 4)
            and not get_layers_by_type(model, qlayers.QuantizedStatefulRecurrent)):
        raise RuntimeError(
            "Input shape of model must be 2-D or 4-D (batch size + 1-D or 3-D "
            f"tensors). Receives input shape {model.input_shape}.")


def _extract_skippable_layers(block):
    """Split block into skippable and non skippable layers

    Args:
        block (keras.Layer): block to split.

    Returns:
        tuple: list of skippable and non skippable layers
    """
    skippable, non_skippable = [], []
    for layer in block:
        if isinstance(layer, skippable_layers):
            skippable.append(layer)
        elif isinstance(layer, qlayers.QuantizedReshape):
            in_shape = layer.input_shape
            out_shape = layer.output_shape
            in_dims = [x for x in in_shape if x != 1]
            out_dims = [x for x in out_shape if x != 1]
            if in_dims != out_dims:
                non_skippable.append(layer)
            else:
                skippable.append(layer)
        else:
            non_skippable.append(layer)
    return skippable, non_skippable
