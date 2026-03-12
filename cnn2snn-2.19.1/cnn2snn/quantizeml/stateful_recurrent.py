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
""" Functions to convert QuantizedStatefulRecurrent to Akida. """

import quantizeml.layers as qlayers
import numpy as np

from akida import StatefulRecurrent, PicoPostProcessing, ActivationType

from ..akida_versions import AkidaVersion
from .activations import set_relu_variables, parse_relu_v2
from .block_converter import BlockConverter, register_conversion_patterns
from .blocks import get_block_out_quantizer
from .conv_common import get_layer_by_type
from .outputs import set_output_v2_variables
from .weights import broadcast_and_set_variable

__all__ = ["StatefulRecurrentBlockConverter"]

_PATTERNS = [(qlayers.QuantizedStatefulProjection, qlayers.QuantizedStatefulRecurrent,
              qlayers.QuantizedExtractToken, qlayers.QuantizedStatefulProjection,
              qlayers.QuantizedReLU),
             (qlayers.QuantizedStatefulProjection, qlayers.QuantizedStatefulRecurrent,
              qlayers.QuantizedExtractToken, qlayers.QuantizedStatefulProjection)]


def _set_stateful_recurrent_variables(ak_layer, block):
    """Computes and sets the variables for an Akida StatefulRecurrent layer.

    This function converts the variables of a Keras layer and sets them into
    the corresponding variables of the equivalent Akida layer.

    Args:
        ak_layer (:obj:`akida.Layer`): the targeted akida layer.
        block (list(:obj:`keras.Layer`)): the block of keras layers.
    """
    variables_ak = ak_layer.variables

    input_proj = block[0]
    stateful_rec = block[1]
    output_proj = block[3]

    assert isinstance(input_proj.weight_quantizer, qlayers.WeightQuantizer)
    assert isinstance(output_proj.weight_quantizer, qlayers.WeightQuantizer)
    assert isinstance(stateful_rec.a_quantizer, qlayers.WeightQuantizer)

    # Set StatefulRecurrent input_shift
    input_shift = getattr(input_proj, 'input_shift', None)
    if input_shift is not None:
        broadcast_and_set_variable(variables_ak, "input_shift",
                                   input_shift.value.numpy().astype(np.uint8))
    # Set input_projection dense kernel
    in_proj_ak = input_proj.weight_quantizer.qweights.value.fp.values.numpy()
    variables_ak["in_proj"] = in_proj_ak.astype(np.int8)

    assert input_proj.out_quantizer is None, "Unexpected out_quantizer on input projection."

    # Set new_state_shift, new_state_scales that is actually recorded in stateful_rec
    if new_state_shift := getattr(stateful_rec, 'new_state_shift', None):
        broadcast_and_set_variable(variables_ak, "new_state_shift",
                                   new_state_shift.value.numpy().astype(np.int8))

    if new_state_scale := getattr(stateful_rec, 'new_state_scale', None):
        broadcast_and_set_variable(variables_ak, "new_state_scale",
                                   new_state_scale.value.values.numpy().astype(np.uint8))

    # Set A weights variables
    A_real_ak = stateful_rec.a_quantizer(stateful_rec.A_real).values.numpy()
    variables_ak["A_real"] = A_real_ak.astype(np.int16)
    A_imag_ak = stateful_rec.a_quantizer(stateful_rec.A_imag).values.numpy()
    variables_ak["A_imag"] = A_imag_ak.astype(np.int16)

    # Set intermediate internal_state_outputs_shift, which is a scale broadcasted to a vector
    internal_state_outputs_shift = stateful_rec.out_quantizer.shift.value.numpy()
    broadcast_and_set_variable(variables_ak, "internal_state_outputs_shift", np.full(
        A_real_ak.shape, internal_state_outputs_shift, dtype=np.int8))

    # Set out_proj dense kernel variable
    out_proj_ak = output_proj.weight_quantizer.qweights.value.fp.values.numpy()
    variables_ak["out_proj"] = out_proj_ak.astype(np.int8)

    if output_proj.use_bias:
        bias_quantizer = output_proj.bias_quantizer
        assert isinstance(bias_quantizer, qlayers.AlignedWeightQuantizer)
        # Set StatefuRecurrent layer bias variable and shift
        bias = bias_quantizer.qweights.value.values.numpy().astype(np.int32)
        bias_shift = bias_quantizer.shift.value.numpy().astype(np.uint8)
        variables_ak["bias"] = (bias >> bias_shift).astype(np.int8)
        broadcast_and_set_variable(variables_ak, "bias_shift", bias_shift)

    # Check if we have ReLU
    relu_layer = get_layer_by_type(block, qlayers.QuantizedReLU)
    # Set optional activation variables
    if relu_layer:
        set_relu_variables(ak_layer, relu_layer)

    # Get the layer block output_quantizer
    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        set_output_v2_variables(ak_layer, out_quantizer)


def _create_stateful_recurrent(block):
    """Parses a QuantizeML quantized stateful recurrent block and returns the
    params to create the corresponding Akida StatefulRecurrent layer.

    Args:
        block (list(:obj:`keras.Layer`)): list of quantized layers.

    Returns:
        :obj:`akida.StatefulRecurrent`: The created Akida layer.
    """
    input_proj = block[0]
    output_proj = block[3]

    # In QuantizeML one bit is reserved automatically for the sign, but in Akida this is rather
    # checked during the clipping operations.
    block_params = dict(
        stateful_channels=input_proj.units,
        output_channels=output_proj.units,
        subsample=output_proj.subsample,
        activation=ActivationType.NoActivation,
        buffer_bits=input_proj.buffer_bitwidth + 1,
    )

    if relu_layer := get_layer_by_type(block, qlayers.QuantizedReLU):
        block_params.update(parse_relu_v2(relu_layer))

    # Specific post_op_buffer_bits and output_bits since last layer can have an activation without
    # an OutputQuantizer
    out_quantizer = get_block_out_quantizer(block)
    if out_quantizer:
        block_params.update({"post_op_buffer_bits": out_quantizer.buffer_bitwidth})
        block_params.update({"output_bits": out_quantizer.bitwidth})
    else:
        block_params.update({"post_op_buffer_bits": 32})

        if get_layer_by_type(block, qlayers.QuantizedReLU) or output_proj.use_bias:
            block_params.update({'output_bits': block_params["post_op_buffer_bits"]})
        else:
            block_params.update({'output_bits': block_params["buffer_bits"]})

    return StatefulRecurrent(**block_params, name=output_proj.name)


def _convert_quantized_stateful_recurrent(model_ak, block, inbound_layers_ak):
    """Converts QuantizedLayerStatefulRecurrent layers block and its variables and adds
    it to the Akida's model.

    The expected sequence is:

    - QuantizedStatefulProjection,
    - QuantizedStatefulRecurrent,
    - QuantizedExtractToken,
    - QuantizedStatefulProjection,
    - QuantizedReLU (optional).

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the model will be added.
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    # Create and add layer to the akida model
    layer_ak = _create_stateful_recurrent(block)
    model_ak.add(layer_ak, inbound_layers_ak)

    # Set the akida layer converted variables
    _set_stateful_recurrent_variables(layer_ak, block)


class StatefulRecurrentBlockConverter(BlockConverter):
    """Main class that should be used to check if the stateful recurrent layer block is compatible
    to an Akida v2 conversion and provides a method to convert it in an equivalent Akida v2
    StatefulRecurrent layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        self._stateful_recurrent_additional_checks()

    def _stateful_recurrent_additional_checks(self):
        input_proj, output_proj = self._block[0], self._block[3]

        # Check input proj has no output quantizer, no bias, no activation
        assert input_proj.out_quantizer is None, "input_proj must not have output quantizer"
        assert not input_proj.use_bias, "input_proj must not use bias"

        # Check buffer bits match between input and output proj
        assert input_proj.buffer_bitwidth == output_proj.buffer_bitwidth, (
            "input_proj and output_proj buffer bits must match")

        # Shaping checks
        assert input_proj.subsample == 1, "input_proj must have subsample=1"

    def convert(self, model_ak, inbounds):
        _convert_quantized_stateful_recurrent(model_ak, self._block, inbounds)


class PicoPostProcessingBlockConverter(BlockConverter):
    def __init__(self, block):
        super().__init__(block)
        self._additional_checks()

    def _additional_checks(self):
        qml_pico = self._block[0]
        if np.any(qml_pico.threshold < 0) or np.any(qml_pico.threshold > 2**16 - 1):
            raise ValueError("PicoPostProcessing threshold must be uint16 values.")

    def convert(self, model_ak, inbounds):
        qml_pico = self._block[0]
        # Create and add layer to the akida model.
        pico = PicoPostProcessing(buffer_bits=qml_pico.buffer_bitwidth + 1, name=qml_pico.name)
        model_ak.add(pico, inbounds)
        # Transfer variables.
        set_output_v2_variables(pico, qml_pico.pred_quantizer, prefix="input")
        broadcast_and_set_variable(pico.variables,
                                   "threshold",
                                   np.array(qml_pico.threshold, dtype=np.uint16))


# Register the valid stateful recurrent block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, StatefulRecurrentBlockConverter)
register_conversion_patterns(AkidaVersion.v2,
                             [(qlayers.QuantizedPicoPostProcessing,)],
                             PicoPostProcessingBlockConverter)
