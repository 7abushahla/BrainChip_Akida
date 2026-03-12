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
"""Functions to convert keras pooling layers parameters to akida.
"""

from quantizeml.layers import QuantizedMaxPool2D
from akida import Padding, PoolType


def parse_max_pool_v1(layer):
    """Parses a quantizeml.QuantizedMaxPool2D parameters for Akida v1.

    Args:
        layer (:obj:`quantizeml.QuantizedMaxPool2D`): the layer to parse.

    Returns:
        dict: the corresponding akida parameters.
    """
    assert isinstance(layer, QuantizedMaxPool2D)

    padding = Padding.Same if layer.padding == "same" else Padding.Valid
    pool_size = layer.pool_size
    pool_stride = layer.strides if layer.strides else pool_size

    pool_params = dict(
        pool_type=PoolType.Max,
        padding=padding,
        pool_size=pool_size,
        pool_stride=pool_stride
    )

    return pool_params


def parse_max_pool_v2(layer):
    """Parses a quantizeml.QuantizedMaxPool2D parameters for Akida v2.

    Args:
        layer (:obj:`quantizeml.QuantizedMaxPool2D`): the layer to parse.

    Returns:
        dict: the corresponding akida parameters.
    """
    assert isinstance(layer, QuantizedMaxPool2D)

    padding = Padding.Same if layer.padding == "same" else Padding.Valid
    pool_size = layer.pool_size
    pool_stride = layer.strides if layer.strides else pool_size

    pool_params = dict(
        pool_type=PoolType.Max,
        padding=padding,
        pool_size=pool_size[0],
        pool_stride=pool_stride[0]
    )

    return pool_params


def set_gap_variables_v1(layer_ak, gap_layer):
    """Computes and sets the pooling variables in an akida v1 layer.

    Args:
        layer_ak (:obj:`akida.Layer`): the targeted akida layer.
        gap_layer (:obj:`quantizeml.Layer`): the source layer.
    """

    # In AkidaVersion.v1, biases are only applied when evaluating activations, by
    # subtracting the threshold. Since this happens after pooling, the scaling factor
    # introduced by the GAP operation must be applied to the threshold.
    spatial_size = gap_layer.input_shape[1] * gap_layer.input_shape[2]
    layer_ak.variables["threshold"] *= spatial_size


def max_pool_param_checks(max_pool_layer):
    """Check if the max_pool_layer has valid parameters for an Akida v2 conversion.
    Raise an error if not.

    Args:
        max_pool_layer (:obj:`akida.Layer`): QuantizedMaxPool2D layer to verify.

    """
    # if max_pool strides are None max_pool.strides == max_pool.pool_size => Valid assertion
    if max_pool_layer.strides is None:
        return

    pool_size = max_pool_layer.pool_size
    pool_stride = max_pool_layer.strides
    if pool_size[0] != pool_size[1]:
        raise ValueError(f"In Akida v2 {max_pool_layer.name} should have square pooling")
    if pool_stride[0] != pool_stride[1]:
        raise ValueError(f"In Akida v2 {max_pool_layer.name} pooling stride should be the same for"
                         "both dimensions")


def gap_params_checks(gap_layer):
    """Check if the gap layer has valid parameters for an Akida v1 conversion.
    Raise an error if not.

    Args:
        gap_layer (:obj:`akida.Layer`): QuantizedGlobalAveragePooling2D layer to verify.

    """

    spatial_size = gap_layer.input_shape[1] * gap_layer.input_shape[2]
    if spatial_size > 2**8:
        raise RuntimeError(f"Unsupported spatial size product ({spatial_size}). "
                           "We only accepts values smaller than 256. "
                           "Consider to replace GlobalAveragePooling2D by MaxPooling2D.")
