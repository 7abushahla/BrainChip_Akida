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
__all__ = ["OnnxConverter"]

import numpy as np
import akida

from onnxruntime.quantization.quant_utils import find_by_name
from quantizeml.onnx_support.layers.base_layer import BRN_OPSET

from . import onnx_graph_tools
from .layer_bounds import get_inbound_layers, get_next_neighbor_nodes
from .weights import load_weights


def get_akida_input_model_shape(onnx_graph, channels_last=False):
    shape = onnx_graph_tools.get_tensor_shape(onnx_graph.input[0])
    # Remove batch size as akida requires a 3D input shape
    akida_shape = shape[1:]
    if len(akida_shape) > 1 and not channels_last:
        # Convert to channels last
        akida_shape = akida_shape[1:] + akida_shape[0:1]
    # Expand to have exactly three dimensions
    akida_shape = np.insert(akida_shape, [0] * (3 - len(akida_shape)), 1)
    return akida_shape


class OnnxConverter:
    """Abstract class that allows to convert a node into the corresponding Akida layer.

    Child should overwrite _additional_checks function (if extra checks are required)
    and implement convert() function.

    Args:
        node (NodeProto): the node to convert.
        model (ModelProto): the model that has the node in it.
    """
    is_input_layer = False
    channels_last = False

    def __init__(self, node, model):
        if node.domain != BRN_OPSET.domain:
            raise ValueError(f"Unrecognized {node.name}: it is not part of the domain.")
        self._node = node
        self._model = model
        self._input_shape = None

        # Parse attributes and weights from node (and initializer) into class
        self.weights = load_weights(node, model.graph.initializer, self.func)
        self.load_attributes(node)

        # Check special rules
        self._additional_checks()

    @property
    def name(self):
        return self._node.name

    @property
    def input_name(self):
        return self._node.input[0]

    @property
    def input_shape(self):
        if self._input_shape is not None and len(self._input_shape) > 0:
            return self._input_shape
        raise AssertionError("Input shape has not been set.")

    @property
    def func(self):
        func = find_by_name(self._node.op_type, self._model.functions)
        assert func, f"{self.name} does not have an associated function."
        return func

    def load_attributes(self, node):
        """Load node attributes into object.

        Args:
            node (NodeProto): the input node.
        """
        for attr in node.attribute:
            # Get attribute value
            value = onnx_graph_tools.get_field(node, attr.name)
            # Set value into class as an attribute
            setattr(self, attr.name, value)

    def _additional_checks(self):
        """Check node compatibility with Akida."""
        if self.is_input_layer and len(get_next_neighbor_nodes(self._node, self._model.graph)) != 1:
            raise RuntimeError("Node must be connected to only one node.")

    def _parse_akida_layer(self):
        raise NotImplementedError("Child must implement this function")

    def _set_akida_variables(self, ak_layer):
        raise NotImplementedError("Child must implement this function")

    def convert(self, ak_model):
        """Convert node into an Akida layer and append it into the model.

        Args:
            ak_model (akida.Model): the target Akida model.
        """
        # Retrieve the akida inbound layers.
        inbound_layers_ak = get_inbound_layers(ak_model, self._node, self._model.graph)

        # Assign input shape.
        input_shape = [x.output_dims for x in inbound_layers_ak]
        if len(input_shape) == 0:
            assert len(ak_model.layers) == 0, "There can be no layers prior."
            input_shape = get_akida_input_model_shape(self._model.graph, self.channels_last)
        elif len(input_shape) == 1:
            input_shape = input_shape[0]
        self._input_shape = input_shape

        # Insert an InputData if the current layer requires it.
        if (not self.is_input_layer and (
            len(inbound_layers_ak) == 0 or
                inbound_layers_ak[0].parameters.layer_type == akida.LayerType.Quantizer)):
            input_data = akida.InputData(input_shape=input_shape, input_bits=8)
            ak_model.add(input_data, inbound_layers_ak)
            # Update inbound layers.
            inbound_layers_ak = [input_data]

        # Declare akida layer to set variables
        ak_layer = self._parse_akida_layer()

        # Append new layer in akida mode
        ak_model.add(ak_layer, inbound_layers_ak)

        # Set weights into akida layer
        try:
            self._set_akida_variables(ak_layer)
        except Exception as e:
            raise type(e)("Impossible to transfer variable. \nReason: " + str(e))
