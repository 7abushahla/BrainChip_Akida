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
"""Utility function to convert layer to akida.
"""
import akida
from tf_keras import layers as keras_layers
from quantizeml import layers as qlayers


def _get_input_data(ak_model):
    for layer in ak_model.layers:
        if layer.parameters.layer_type == akida.LayerType.InputData:
            return layer
    return None


def get_inbound_layers(ak_model, k_layer):
    """Returns a list of Akida layers in an Akida model that correspond to the inbound
    layers of a Keras layer. The list will be built by traversing the Keras model graph.

    Note: Only keras layers that have a corresponding layer in the akida model can be found.
    e.g. InputData cannot be found, since it has no representation in the keras model.

    Args:
        ak_model (:obj:`Model`): the model where to find the inbounds layers.
        k_layer (:obj:`keras.Layer`): the source layer.

    Returns:
        :list: the inbounds layers names.
    """

    # The list of supported Keras layers not represented in Akida
    skippable_layers = (qlayers.QuantizedDropout, qlayers.QuantizedFlatten,
                        qlayers.QuantizedReshape, qlayers.QuantizedRescaling,
                        qlayers.QuantizedReLU, qlayers.QuantizedMaxPool2D,
                        qlayers.QuantizedGlobalAveragePooling2D,
                        qlayers.QuantizedActivation, keras_layers.InputLayer)

    # Get inbound layers names
    in_node = k_layer.inbound_nodes[0]
    inbound_layers = in_node.inbound_layers
    if not isinstance(inbound_layers, list):
        inbound_layers = [inbound_layers]
    non_skippable_inbounds_names = []
    for layer in inbound_layers:
        if type(layer) in skippable_layers:
            # All skippable layers have one inbound layer, so this can be
            # easily resolved.
            while type(layer) in skippable_layers:
                layer = layer.inbound_nodes[0].inbound_layers
        # If all layers have been skipped, the inbound is either an InputData or None.
        if layer == []:
            if input_data := _get_input_data(ak_model):
                non_skippable_inbounds_names.append(input_data.name)
            elif len(inbound_layers) > 1:
                raise RuntimeError(f"Cannot resolve inbound layers for {k_layer.name}.")
        else:
            non_skippable_inbounds_names.append(layer.name)
    return [ak_model.get_layer(lname) for lname in non_skippable_inbounds_names]
