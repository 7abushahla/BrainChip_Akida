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
__all__ = ["AddOnnxConverter"]

import numpy as np

import akida

from .base_converter import OnnxConverter
from .register import register_onnx_converter_target
from .weights import broadcast_and_set_variable
from .scale_out import set_output_scale_variables


def set_input_shift_variable(ak_layer, shift, var_name="input_shift"):
    # Shift integrity check
    ak_shift = np.array(np.log2(shift), "uint8")
    if np.any(2.0**ak_shift != shift) or np.any(shift == 0):
        raise ValueError(f"Error found when setting shift in {ak_layer.name}:",
                         f"{shift} is not a power of two.")
    # Set variable
    broadcast_and_set_variable(ak_layer.variables, var_name, ak_shift)


@register_onnx_converter_target("QuantizedAdd")
class AddOnnxConverter(OnnxConverter):
    """Convert QuantizedAdd type node into akida.Add.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model containing the node.
    """

    def load_attributes(self, node):
        # Some attributes should infer from node.op_type
        n_op = node.op_type
        self.output_bits = 8 if "Scaled" in n_op else 32
        return super().load_attributes(node)

    def _additional_checks(self):
        super()._additional_checks()
        # Reject Add(X, X) (inputs are the same tensor)
        if self._node.input[0] == self._node.input[1]:
            raise RuntimeError("Inputs must come from different tensors.")

    def _parse_akida_layer(self):
        return akida.Add(output_bits=self.output_bits, name=self.name)

    def _set_akida_variables(self, ak_layer):
        assert isinstance(ak_layer, akida.Add)

        # Input shifts
        set_input_shift_variable(ak_layer, self.weights["Xs"], "a_shift")
        set_input_shift_variable(ak_layer, self.weights["Ys"], "b_shift")

        # Shift out
        set_output_scale_variables(ak_layer, shift=self.weights.get("Shift", 1))
