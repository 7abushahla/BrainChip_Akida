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
__all__ = ["get_inbound_layers", "get_next_neighbor_nodes"]


def get_inbound_layers(ak_model, node, graph):
    """Returns a list of Akida layers in an Akida model that correspond to the inbound
    nodes of a NodeProto.

    Args:
        ak_model (Model): the model where to find the inbounds layers.
        node (NodeProto): the node to get the inbounds.
        graph (GraphProto): the graph containing the node.

    Returns:
        list: the inbounds akida layers.
    """
    # Get the nodes connected to the inputs, keeping the same order
    inbound_nodes = [""] * len(node.input)
    inodes_names = list(node.input)
    for t_node in graph.node:
        try:
            inbound_nodes[inodes_names.index(t_node.output[0])] = t_node
        except ValueError:
            pass
    inbound_nodes = list(filter(lambda x: x, inbound_nodes))

    # Check each inbound has a name
    if not all([node.name for node in inbound_nodes]):
        raise ValueError("All inbound nodes must have a name.")

    # Get the corresponding Akida layers
    return [ak_model.get_layer(node.name) for node in inbound_nodes]


def get_next_neighbor_nodes(node, graph):
    # Get the nodes connected to the outputs
    outbounds = []
    for target_node in graph.node:
        for target_input in target_node.input:
            if target_input in node.output:
                outbounds.append(target_node)
    return outbounds
