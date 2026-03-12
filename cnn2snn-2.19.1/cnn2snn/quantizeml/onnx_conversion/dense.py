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
__all__ = ["Dense1DOnnxConverter"]

import numpy as np

import akida

from .base_converter import OnnxConverter
from .register import register_onnx_converter_target
from .weights import set_weight_variables
from .activation import set_relu_variables, parse_activation_type
from .scale_out import set_output_scale_variables


@register_onnx_converter_target("QuantizedDense1D")
class Dense1DOnnxConverter(OnnxConverter):
    """Convert QuantizedDense1D type node into akida.Dense1D.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model that the node is.
    """

    def load_attributes(self, node):
        # Some attributes should infer from node.op_type
        n_op = node.op_type
        self.activation = parse_activation_type(n_op)
        self.output_bits = 8 if "Scaled" in n_op else 32
        return super().load_attributes(node)

    def _parse_akida_layer(self):
        # Parse common information
        layer_params = {
            "name": self.name,
            "units": self.weights["W"].shape[0],
            "output_bits": self.output_bits,
            "activation": self.activation,
        }
        return akida.Dense1D(**layer_params)

    def _set_akida_variables(self, ak_layer):
        assert isinstance(ak_layer, akida.Dense1D)

        # Reshape kernel in the format akida expects (even if the operation comes from a flatten)
        # Note: original input_shape is (c, x, y), but property return akida shape
        x, y, c = self.input_shape
        # First, unroll flattened inputs
        kernel = np.reshape(self.weights["W"], (-1, c, x, y))
        # Second, transpose to match akida ordering
        kernel = np.transpose(kernel, (2, 3, 1, 0))
        # Finally, flatten again
        kernel = np.reshape(kernel, (x * y * c, -1))

        # Get bias
        bias = self.weights.get("bias", None)
        set_weight_variables(ak_layer, kernel, bias)

        # Activation
        if self.activation == akida.ActivationType.ReLU:
            set_relu_variables(ak_layer, self.weights.get("max_value", None))

        # Scale out
        set_output_scale_variables(ak_layer,
                                   self.weights.get("Scale", 1),
                                   self.weights.get("Shift", 1))
