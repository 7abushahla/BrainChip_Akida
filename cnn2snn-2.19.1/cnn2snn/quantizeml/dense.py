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
"""Functions to convert QuantizedDense to Akida.
"""
from akida import Dense1D, FullyConnected, ActivationType
from quantizeml.layers import (QuantizedDense, QuantizedReLU, QuantizedReshape, QuantizedFlatten,
                               WeightQuantizer, AlignedWeightQuantizer)
import numpy as np

from .weights import broadcast_and_set_variable
from ..akida_versions import AkidaVersion
from .activations import parse_relu_v1, parse_relu_v2, set_relu_variables, v1_relu_checks
from .outputs import (set_output_v1_variables, set_output_v2_variables, parse_output_bits,
                      parse_post_op_buffer_bits)
from .blocks import get_block_out_quantizer
from .block_converter import BlockConverter, register_conversion_patterns

__all__ = ["DenseBlockConverterV1", "DenseBlockConverterV2"]


_PATTERNS = [(QuantizedDense,), (QuantizedDense, QuantizedReLU), (QuantizedFlatten, QuantizedDense),
             (QuantizedFlatten, QuantizedDense, QuantizedReLU), (QuantizedReshape, QuantizedDense),
             (QuantizedReshape, QuantizedDense, QuantizedReLU),
             (QuantizedReshape, QuantizedFlatten, QuantizedDense),
             (QuantizedReshape, QuantizedFlatten, QuantizedDense, QuantizedReLU)]


def _set_dense_v1_variables(ak_layer, dense, block_input_shape):
    """Computes and sets the variables for an Akida FullyConnected layer.

    This function converts the variables of a dense block of Keras layers and sets them into
    the corresponding variables of the equivalent Akida FullyConnected layer.

    Args:
        ak_layer (:obj:`akida.Layer`): the targeted akida layer.
        dense (:obj:`keras.Layer`): the keras dense layer.
        block_input_shape (list): the dense block input shape.
    """
    assert isinstance(dense.weight_quantizer, WeightQuantizer)
    variables_ak = ak_layer.variables

    # get the QuantizedDense weights
    weights_ak = dense.weight_quantizer.qweights.value.fp.values.numpy()
    # get the QuantizedDense bias and shift
    if dense.use_bias:
        bias_quantizer = dense.bias_quantizer
        assert isinstance(bias_quantizer, AlignedWeightQuantizer)
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        # Store bias into the threshold variable
        variables_ak["threshold"] = -bias

    # In AkidaVersion.v1, it is only possible to convert flat inputs
    input_shape = dense.input_shape
    units = ak_layer.parameters.units
    if block_input_shape != input_shape:
        _, X, Y, C = block_input_shape
        # When a fully connected layer follows a convolutional layer, we need to modify
        # the way the weights are laid out because the Keras and Akida flatten operations
        # that happen on spatial dimensions are inverted (row-major versus col-major).
        # We therefore need to:
        # - reshape the Keras (FxN) weights to match the block input shape (XxYxCxN),
        # - transpose to the equivalent akida shape (CxYxXxN).
        weights_ak = weights_ak.reshape(X, Y, C, units).transpose(2, 1, 0, 3)
    weights_ak = weights_ak.reshape(1, 1, input_shape[-1], units)
    variables_ak["weights"] = weights_ak.astype(np.int8)


def _set_dense_v2_variables(ak_layer, dense):
    """Computes and sets the variables for an Akida Dense1 layer.

    This function converts the variables of a dense block of Keras layers and sets them into
    the corresponding variables of the equivalent Akida Dense1D layer.

    Args:
        ak_layer (:obj:`akida.Layer`): the targeted akida layer.
        dense (:obj:`keras.Layer`): the keras dense layer.
    """
    assert isinstance(dense.weight_quantizer, WeightQuantizer)
    variables_ak = ak_layer.variables

    # get the QuantizedDense weights
    weights_ak = dense.weight_quantizer.qweights.value.fp.values.numpy()
    # get the QuantizedDense bias and shift
    if dense.use_bias:
        bias_quantizer = dense.bias_quantizer
        assert isinstance(bias_quantizer, AlignedWeightQuantizer)
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        bias_shift = bias_quantizer.shift.value.numpy().astype(np.uint8)
        variables_ak["bias"] = (bias >> bias_shift).astype(np.int8)
        broadcast_and_set_variable(variables_ak, "bias_shift", bias_shift)

    input_shift = getattr(dense, 'input_shift', None)
    if input_shift is not None:
        broadcast_and_set_variable(variables_ak, "input_shift",
                                   input_shift.value.numpy().astype(np.uint8))

    variables_ak["weights"] = weights_ak.astype(np.int8)


def _parse_dense_v1(layer):
    """Parses a quantizeml.QuantizedDense parameters that target the akida v1 FullyConnected layer.

    Args:
        layer (:obj:`keras.Layer`): the quantizeml.QuantizedDense layer to parse.

    Returns:
        dict: the corresponding akida parameters.

    """
    # Set by default the activation to false. It will be updated when parsing the optional ReLU
    # layer.
    dense_params = dict(units=layer.units, activation=False, name=layer.name)

    weight_bits = layer.weight_quantizer.bitwidth
    if weight_bits not in [1, 2, 4]:
        raise ValueError(f"FullyConnected layer only supports 1, 2 or 4 weight bits value. \
                              Received weight_bits={weight_bits}.")
    dense_params["weights_bits"] = weight_bits

    return dense_params


def _parse_dense_v2(layer):
    """Parses a quantizeml.QuantizedDense parameters that target the akida v2 Dense1D
    layer.

    Args:
        layer (:obj:`keras.Layer`): the quantizeml.QuantizedDense layer to parse.

    Returns:
        dict: the corresponding akida parameters.

    """
    # In quantizeml one bit is reserved for the sign in the buffer bitwidth
    # variable, but in akida this value has to be added back to have the
    # correct clipping.
    buffer_bits = layer.buffer_bitwidth + 1

    # Set by default the activation to false. It will be updated when parsing the optional ReLU
    # layer.
    dense_params = dict(units=layer.units, activation=ActivationType.NoActivation, name=layer.name)

    dense_params["buffer_bits"] = buffer_bits

    weight_bits = layer.weight_quantizer.bitwidth

    dense_params["weights_bits"] = weight_bits
    return dense_params


def _get_dense_layer(block):
    """Helper to extract QuantizedDense layer of the block.

    Args:
        block (list(:obj:`keras.Layer`)): the layers block to convert.

    Return:
        :obj:`keras.Layer`: the QuantizedDense layer of the block
    """
    for layer in block:
        if isinstance(layer, QuantizedDense):
            return layer
    raise RuntimeError("No QuantizedDense found in that block."
                       f" Received: {[l.__class__.__name__ for l in block]}")


def convert_block_to_akida_dense_layer(model_ak, block, akida_type, inbound_layers_ak):
    """Converts a dense block into an akida dense layer target.

    The expected sequence is:
        - QuantizedFlatten/QuantizedReshape (optional),
        - QuantizedDense,
        - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the layer will be added.
        block (list(:obj:`keras.Layer`)): the layers block to convert.
        akida_type (:obj:`akida.LayerType`): the targeted akida layer type.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """

    # Extract dense layer of the block
    dense = _get_dense_layer(block)

    # Parse main dense params
    if akida_type == FullyConnected:
        dense_params = _parse_dense_v1(dense)
    else:
        dense_params = _parse_dense_v2(dense)
        # parse the block output bits
        parse_output_bits(block, dense_params)
        # parse the block post op buffer bits
        parse_post_op_buffer_bits(block, dense_params)

    # Extract ReLU if present
    relu_layer = None
    if isinstance(block[-1], QuantizedReLU):
        relu_layer = block[-1]
        # parse ReLU layer parameters
        if akida_type == FullyConnected:
            act_params = parse_relu_v1(relu_layer)
        else:
            act_params = parse_relu_v2(relu_layer)
        dense_params.update(act_params)

    # Create the Akida layer
    dense_ak = akida_type(**dense_params)
    model_ak.add(dense_ak, inbound_layers_ak)

    # Set the main Akida dense layer variables depending on the layer type
    if akida_type == FullyConnected:
        _set_dense_v1_variables(dense_ak, dense, block[0].input_shape)
    else:
        _set_dense_v2_variables(dense_ak, dense)
        # Set the optional activation variables for v2 layers:
        if relu_layer:
            set_relu_variables(dense_ak, relu_layer)

    # Get out_quantizer of the block.
    out_quantizer = get_block_out_quantizer(block)

    # Set the optional output_quantizer variables
    if out_quantizer:
        set_output_v1_variables(dense_ak, out_quantizer) if akida_type == FullyConnected else\
            set_output_v2_variables(dense_ak, out_quantizer)


class DenseBlockConverterV1(BlockConverter):
    """Main class that should be used to check if the dense block is compatible to an Akida v1
    conversion and provides a method to convert it in an equivalent Akida FullyConnected layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._dense_layer_checks()
        # Check ReLU compatibility for v1 only
        if isinstance(block[-1], QuantizedReLU):
            v1_relu_checks(block[-1])

    def convert(self, model_ak, inbounds):
        convert_block_to_akida_dense_layer(model_ak, self._block, FullyConnected, inbounds)

    def _dense_layer_checks(self):
        dense = _get_dense_layer(self._block)
        # Get dense layer input shape without the batch size.
        dense_shape = dense.input_shape[1:]
        # For a shape (X, Y, Z), if X*Y*Z==Z <=> X=Y=1 <=> the shape is flattened.
        if not (np.prod(dense_shape) == dense_shape[-1]):
            raise RuntimeError("The dense layer only handles flattened shapes"
                               f" (i.e (1, 1, X) or (X,)). Received: ({dense.input_shape[1:]}) "
                               " shape.")


class DenseBlockConverterV2(BlockConverter):
    """Main class that should be used to check if the dense block is compatible to an Akida v2
    conversion and provides a method to convert it in an equivalent Akida Dense1D layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._dense_layer_checks()

    def convert(self, model_ak, inbounds):
        convert_block_to_akida_dense_layer(model_ak, self._block, Dense1D, inbounds)

    def _dense_layer_checks(self):
        dense = _get_dense_layer(self._block)
        # Get dense layer input shape without the batch size.
        dense_shape = dense.input_shape[1:]
        # For a shape (X, Y, Z), if X*Y*Z==Z <=> X=Y=1 <=> the shape is flattened.
        if not (np.prod(dense_shape) == dense_shape[-1]):
            raise RuntimeError("The dense layer only handles flattened shapes"
                               f" (i.e (1, 1, X) or (X,)). Received: ({dense.input_shape[1:]}) "
                               " shape.")


# Register the valid dense block patterns for Akida v1 and v2
register_conversion_patterns(AkidaVersion.v1, _PATTERNS, DenseBlockConverterV1)
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, DenseBlockConverterV2)
