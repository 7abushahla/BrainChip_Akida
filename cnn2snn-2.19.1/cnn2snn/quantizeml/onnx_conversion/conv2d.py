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
__all__ = ["Conv2DOnnxConverter", "InputConv2DOnnxConverter"]

import akida

from .base_converter import OnnxConverter
from .register import register_onnx_converter_target
from .conv_commons import (check_convolution_compatibility, parse_convolutional_parameters,
                           set_convolutional_variables)
from .activation import parse_activation_type


@register_onnx_converter_target("QuantizedConv2D")
class Conv2DOnnxConverter(OnnxConverter):
    """Convert QuantizedConv2D type node into an akida.Conv2D.

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
        if "GlobalAvgPool" in n_op:
            self.pool_type = akida.PoolType.Average
        elif "MaxPool" in n_op:
            self.pool_type = akida.PoolType.Max
        else:
            self.pool_type = akida.PoolType.NoPooling

        # Padding type is inferred from pads attribute
        self.pads = self.weights["pads"].tolist()

    def _additional_checks(self):
        check_convolution_compatibility(self)

    def _parse_akida_layer(self):
        layer_params = parse_convolutional_parameters(self)
        return akida.Conv2D(**layer_params)

    def _set_akida_variables(self, ak_layer):
        set_convolutional_variables(self, ak_layer)


@register_onnx_converter_target("QuantizedInputConv2D")
class InputConv2DOnnxConverter(Conv2DOnnxConverter):
    """Convert QuantizedInputConv2D type node into an akida.InputConv2D.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model that the node is.
    """
    is_input_layer = True

    def _parse_akida_layer(self):
        layer_params = parse_convolutional_parameters(self)
        return akida.InputConv2D(input_shape=self.input_shape, **layer_params)

    def _set_akida_variables(self, ak_layer):
        set_convolutional_variables(self, ak_layer, flip=False)
