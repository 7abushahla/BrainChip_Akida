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
__all__ = ["generate_onnx_model"]
import akida
import warnings
from onnx.checker import check_model

from .register import map_node_to_converter


def generate_onnx_model(model):
    """Generates an Akida model based on an ONNX quantizeml model.

    Args:
        model (obj:`onnx.ModelProto`): a ONNX model to convert.

    Returns:
        akida.Model: the generated Akida model.
    """
    # Model must be compatible with ONNX.
    check_model(model, full_check=True)

    # Checks over model.
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Cannot convert: model must have exactly one input and one output.")

    # Main conversion loop.
    akida_model = akida.Model()
    for idx, node in enumerate(model.graph.node):
        try:
            converter = map_node_to_converter(node, model)
            converter.convert(akida_model)
            if node.op_type == "Dequantizer":
                # End conversion at the first dequantizer found.
                idx += 1
                break
        except Exception as e:
            raise type(e)(f"Cannot convert {node.name}: {e}") from e

    # Check model is not empty (at least one node to convert is required).
    if akida.graph_utils.first_mappable_layer(akida_model.layers) is None:
        raise ValueError("Model is empty or does not have any node to convert.")

    # Prints a warning message with a summary of the non convertible nodes
    skip_nodes = model.graph.node[idx:]
    if len(skip_nodes) > 0:
        stop_layer_msg = " at node " + model.graph.node[idx].name
        skip_layers_summary = "___________________________________________________\n"
        skip_layers_summary += "Node (type)\n"
        skip_layers_summary += "===================================================\n"
        for node in skip_nodes:
            skip_layers_summary += node.name + " (" + node.op_type + ")\n"
        skip_layers_summary += "===================================================\n"
        warnings.warn(f"Conversion stops {stop_layer_msg} because of a dequantizer. "
                      f"The end of the graph is ignored:\n {skip_layers_summary}. \n This can be "
                      "expected for model heads (e.g. softmax for classification) but could also "
                      "mean that processing layers were not quantized.")
    return akida_model
