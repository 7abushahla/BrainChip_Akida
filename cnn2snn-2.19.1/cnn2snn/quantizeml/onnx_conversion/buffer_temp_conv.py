#!/usr/bin/env python
# ******************************************************************************
# Copyright 2025 Brainchip Holdings Ltd.
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
__all__ = ["BufferTempConvOnnxConverter"]

import akida

from .base_converter import OnnxConverter
from .register import register_onnx_converter_target
from .weights import set_weight_variables
from .scale_out import set_output_scale_variables
from .activation import set_relu_variables, parse_activation_type
from .conv_commons import _check_if_squared


@register_onnx_converter_target("QuantizedBufferTempConv")
class BufferTempConvOnnxConverter(OnnxConverter):
    """Convert QuantizedBufferTempConv type node into akida.BufferTempConv.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model containing the node.
    """
    @property
    def filters(self):
        return self.weights["W"].shape[0]

    def load_attributes(self, node):
        # Some attributes should infer from node.op_type
        n_op = node.op_type
        self.output_bits = 8 if "Scaled" in n_op else 32
        self.buffer_bits = 32
        self.post_op_buffer_bits = 32
        self.activation = parse_activation_type(n_op)
        return super().load_attributes(node)

    def _additional_checks(self):
        super()._additional_checks()
        kernel_shapes = self.weights["W"].shape[-2:]
        _check_if_squared(kernel_shapes, name=f"{self.name} kernels")

    def _parse_akida_layer(self):
        return akida.BufferTempConv(
            filters=self.filters,
            fifo_size=self.fifo_size,
            activation=self.activation,
            output_bits=self.output_bits,
            buffer_bits=self.buffer_bits,
            post_op_buffer_bits=self.post_op_buffer_bits,
            name=self.name)

    def _set_akida_variables(self, ak_layer):
        # Since the weights of the FifoOp conv are stored differently in quantizeml (F, C * T, H, W)
        # vs (H, W, T * C, F), transpose + reshape ops are performed to reorder the
        # (temporal/channels) dimensions of the fifo and to align the weights with akida.
        F, M, H, W = self.weights["W"].shape
        C = M // self.fifo_size
        kernel = self.weights["W"].reshape((F, C, self.fifo_size, H, W))
        kernel = kernel.transpose((3, 4, 2, 1, 0))
        kernel = kernel.reshape((H, W, M, F))
        ak_layer.variables["weights"] = kernel

        bias = self.weights.get("bias", None)
        set_weight_variables(ak_layer, kernel, bias)

        if self.activation == akida.ActivationType.ReLU:
            set_relu_variables(ak_layer, self.weights.get("max_value", None))

        # Scale out
        set_output_scale_variables(ak_layer, self.weights["Scale"], self.weights["Shift"])


@register_onnx_converter_target("QuantizedDepthwiseBufferTempConv")
class DepthwiseBufferTempConvOnnxConverter(BufferTempConvOnnxConverter):
    """Convert QuantizedDepthwiseBufferTempConv type node into akida.DepthwiseBufferTempConv.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model containing the node.
    """

    def _parse_akida_layer(self):
        return akida.DepthwiseBufferTempConv(
            fifo_size=self.fifo_size,
            activation=self.activation,
            output_bits=self.output_bits,
            buffer_bits=self.buffer_bits,
            post_op_buffer_bits=self.post_op_buffer_bits,
            name=self.name)
