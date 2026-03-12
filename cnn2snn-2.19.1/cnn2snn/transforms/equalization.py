#!/usr/bin/env python
# ******************************************************************************
# Copyright 2021 Brainchip Holdings Ltd.
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
"""Equalization tools for Keras/CNN2SNN Sequential models.

   These transformations take models with heterogeneities in their weights or
   biases that are identified as harmful for the model quantization of for their
   conversion to akida. They produce models with reduced heterogeneities, yet
   returning nearly identical results.
"""

import numpy as np
import tf_keras as keras

from .clone import clone_model_with_weights


def _get_filter_max_values(layer):
    if isinstance(layer, keras.layers.Conv2D):
        # Layer "weights" are filters, biases
        filters = layer.get_weights()[0]
        # Filters are HWCN, i.e. the filter index is the last dimension
        return np.amax(np.abs(filters), axis=(0, 1, 2))
    if isinstance(layer, keras.layers.SeparableConv2D):
        # Layer "weights" are depthwise filters, pointwise filters, biases
        # We are only interested in rescaling pointwise filters
        filters = layer.get_weights()[1]
        # Pointwise filters are HWCN, i.e. the filter index is the last
        # dimension
        return np.amax(np.abs(filters), axis=(0, 1, 2))
    if isinstance(layer, keras.layers.Dense):
        # Layer "weights" are filters, biases
        filters = layer.get_weights()[0]
        # Filters are CN, i.e. the filter index is the last dimension
        return np.amax(np.abs(filters), axis=0)
    return None


def _get_channel_max_values(layer):
    if isinstance(layer, keras.layers.Conv2D):
        # Layer "weights" are filters, biases
        filters = layer.get_weights()[0]
        # Filters are HWCN, i.e. the channel index is the third dimension
        return np.amax(np.abs(filters), axis=(0, 1, 3))
    if isinstance(layer, keras.layers.SeparableConv2D):
        # Layer "weights" are depthwise filters, pointwise filters, biases
        dw_filters = layer.get_weights()[0]
        # Depthwise filters are HWCN, i.e. channel index is the third dimension
        return np.amax(np.abs(dw_filters), axis=(0, 1, 3))
    if isinstance(layer, keras.layers.Dense):
        # Layer "weights" are filters, biases
        filters = layer.get_weights()[0]
        # Filters are CN, i.e. the channel index is the first dimension
        return np.amax(np.abs(filters), axis=1)
    return None


def _rescale_fused_layer(layer, scales):
    weights = layer.get_weights()
    if isinstance(layer, keras.layers.SeparableConv2D):
        # Layer "weights" are depthwise filters, pointwise filters, biases
        dw_filters = weights[0]
        pw_filters = weights[1]
        # Divide dw_filters and multiply pw_filters by the same scale
        for i in range(scales.shape[0]):
            dw_filters[:, :, i, :] = dw_filters[:, :, i, :] / scales[i]
            pw_filters[:, :, i, :] = pw_filters[:, :, i, :] * scales[i]
        weights[0] = dw_filters
        weights[1] = pw_filters
        # Update layer weights
        layer.set_weights(weights)


def _get_homogeneity_rate(layer):
    filter_max = _get_filter_max_values(layer)
    if filter_max is None:
        return None
    global_max = np.max(filter_max)
    # Evaluate the relative score of each filter
    filter_rate = filter_max / global_max
    return np.mean(filter_rate)


def weights_homogeneity(model):
    """Give an estimation of the homogeneity of layer weights

    For each Conv or Dense layer in the model, this compares the ranges of
    the weights for each filter with the range of the tensor.
    The score for each filter is expressed as an homogeneity rate (1 is the
    maximum), and the layer homogeneity rate is the mean of all filter rates.

    Args:
        model (:obj:`keras.Model`): a Keras model.
    Returns:
        dict: rates indexed by layer names.
    """
    scores = {}
    for layer in model.layers:
        score = _get_homogeneity_rate(layer)
        if score is not None:
            scores[layer.name] = score
    return scores


def normalize_separable_layer(layer):
    """This normalizes the depthwise weights of a SeparableConv2D.

       In order to limit the quantization error when using a per-tensor
       quantization of depthwise weights, this rescales all depthwise weights
       to fit within the [-1, 1] range.
       To preserve the output of the layer, each depthwise kernel is rescaled
       independently to the [-1, 1] interval by dividing all weights by the
       absolute maximum value, and inversely, all pointwise filters 'looking'
       at these kernels are multiplied by the same value.

    Args:
        layer (:obj:`keras.layers.SeparableConv2D`): a Keras SeparableConv2D
            layer.
    """
    if not isinstance(layer, keras.layers.SeparableConv2D):
        raise ValueError("The layer is not a SeparableConv2D")
    # Get maximum ranges per channel
    dw_max = _get_channel_max_values(layer)
    # Rescale depthwise to [-1, 1] and adjust pointwise accordingly
    _rescale_fused_layer(layer, dw_max)


def normalize_separable_model(model):
    """This normalizes the depthwise weights of all SeparableConv2D in a Model.

    Args:
        model (:obj:`keras.Model`): a Keras model.

    Returns:
        :obj:`keras.Model`: a new Keras model with normalized depthwise
        weights in SeparableConv2D layers.
    """
    # Clone model
    new_model = clone_model_with_weights(model)

    # Normalize SeparableConv2D depthwise weights
    for layer in new_model.layers:
        if isinstance(layer, keras.layers.SeparableConv2D):
            normalize_separable_layer(layer)

    return new_model
