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
"""Utility function referent to padding.
"""
import numpy as np

import quantizeml

from ..model_generator import _get_padding

quantized_conv = (quantizeml.layers.QuantizedConv2D,
                  quantizeml.layers.QuantizedDepthwiseConv2D,
                  quantizeml.layers.QuantizedSeparableConv2D)


def get_padding(k_layer):
    """From a convolution layer, return the akida.Padding.

    Args:
        k_layer (:obj:`keras.Layer`): the source quantized layer.

    Returns:
        tuple: ``akida.Padding``.
    """
    assert isinstance(k_layer, quantized_conv)

    padding = k_layer.padding
    # If a padding_value value is provided or layer produces same output with 'same' or 'valid',
    # force the padding to be "same" in Akida.
    if (getattr(k_layer, "padding_value", None) is not None
            or check_same_valid_compatibility(k_layer)):
        padding = "same"
    return _get_padding(padding)


def get_padding_value(k_layer):
    """From a convolution layer, return the padding value.

    Args:
        k_layer (:obj:`keras.Layer`): the source quantized layer.

    Returns:
        tuple: ``akida.Padding`` and the array of padding values.
    """
    # Padding value must be built in constructor
    padding_ak_value = np.array(0, "int32")
    if getattr(k_layer, "padding_value", None) is not None:
        padding_quantize = k_layer.padding_value_quantizer
        assert isinstance(padding_quantize, quantizeml.layers.AlignedWeightQuantizer)
        padding_ak_value = padding_quantize.qweights.value.values.numpy().astype(np.int32)
    if np.any(padding_ak_value < 0) or np.any(padding_ak_value > 255):
        raise ValueError("padding_value must be an uint8 value")
    return padding_ak_value


def check_same_valid_compatibility(layer):
    """Check if a layer produces the same output regardless of its padding ('same' or 'valid').

    Args:
        layer (:obj:`akida.Layer`): Layer to verify

    Returns:
        bool: same/valid compatibility result.
    """
    # Layer produces same output when kernel size == 1
    return layer.kernel_size == (1, 1)


def check_conv_and_max_pool_compatibility(conv_layer, max_pool_layer):
    """Check if a conv layer and the following max_pool_layer are compatible for a conversion.
    Raise an error if not.

    Args:
        conv_layer (:obj:`akida.Layer`): QuantizedConv2D layer to verify.
        max_pool_layer (:obj:`akida.Layer`): QuantizedMaxPool2D layer to verify.

    """

    neur_pad = conv_layer.padding if not check_same_valid_compatibility(conv_layer) else "same"
    pool_pad = getattr(max_pool_layer, "padding", "")
    if neur_pad != pool_pad:
        raise ValueError(f"Pooling layer {max_pool_layer.name} (padding: {pool_pad}) must"
                         f" have the same padding as {conv_layer.name} (padding: {neur_pad}).")
