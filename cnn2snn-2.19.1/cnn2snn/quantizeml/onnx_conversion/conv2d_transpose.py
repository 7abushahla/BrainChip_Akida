#!/usr/bin/env python
# ******************************************************************************
# Copyright 2024 Brainchip Holdings Ltd.
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
__all__ = ["Conv2DTransposeOnnxConverter"]

import akida

from .activation import parse_activation_type
from .base_converter import OnnxConverter
from .conv_commons import (parse_convolutional_parameters, set_convolutional_variables,
                           check_convolution_compatibility)
from .padding import compute_conv_transpose_same_pads
from .register import register_onnx_converter_target


@register_onnx_converter_target("QuantizedConv2DTranspose")
class Conv2DTransposeOnnxConverter(OnnxConverter):
    """Convert QuantizedConv2DTranspose type node into an akida.Conv2DTranspose.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model that the node is.
    """

    def load_attributes(self, node):
        # Load default attributes
        super().load_attributes(node)

        # Some attributes should infer from node.op_type
        n_op = node.op_type
        self.activation = parse_activation_type(n_op)

    def _additional_checks(self):
        super()._additional_checks()
        check_convolution_compatibility(self)

        # The only supported strides in akida is (2, 2).
        if self.strides != [2, 2]:
            raise ValueError(f"{self.name} expects to have strides equal to (2, 2), "
                             f"found '{self.strides}'.")

        # Check padding is SAME
        kernel_shapes = self.weights["W"].shape[-2:]
        expected_padding = compute_conv_transpose_same_pads(kernel_shapes, self.strides)
        if self.pads != expected_padding:
            raise ValueError(f"{self.name} expects to have '{expected_padding}' pads, "
                             f"found '{self.pads}'.")

    def _parse_akida_layer(self):
        layer_params = parse_convolutional_parameters(self, transpose=True)
        return akida.Conv2DTranspose(**layer_params)

    def _set_akida_variables(self, ak_layer):
        set_convolutional_variables(self, ak_layer, flip=False, transpose=True)
