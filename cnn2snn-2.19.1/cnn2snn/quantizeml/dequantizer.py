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
"""Functions to convert Dequantizer to Akida.
"""
import numpy as np

import akida
from quantizeml.layers import Dequantizer

from .weights import broadcast_and_set_variable
from .block_converter import BlockConverter, register_conversion_patterns
from ..akida_versions import AkidaVersion


__all__ = ["DequantizerBlockConverterV1", "DequantizerBlockConverterV2"]

_PATTERNS = [(Dequantizer,)]


def _get_dequantizer_scales(dequantizer):
    """Helper to extract the common Dequantizer layer scale variable.

    Args:
        dequantizer (:obj:`keras.Layer`): the keras Dequantizer layer.

    Returns:
        ndarray: the layer scales in a numpy float32 array format.

    """
    assert isinstance(dequantizer, Dequantizer)

    # Extract the Dequantizer variables
    frac_bits = dequantizer.frac_bits
    scales = dequantizer.scales
    if isinstance(frac_bits, (list, tuple)):
        frac_bits = frac_bits[0]
        scales = scales[0] if scales else None

    # We project the frac_bits into the scales as:
    #   new_scales = scales / 2 ** frac_bits
    scales = scales.value if scales else 1.0
    scales /= 2**frac_bits.value
    scales = scales.numpy().astype(np.float32)

    return scales


def _convert_dequantizer_v1(model_ak, block):
    """Converts Dequantizer layer and set its variables into the Akida v1 model last layer.

    Args:
        model_ak (:obj:`akida.Model`): the model where the layer will be added.
        block (list(:obj:`keras.Layer`)): the block of keras layers.
    """

    # Take the last layer to set variables
    layer_ak = model_ak.layers[-1]

    if layer_ak.parameters.activation != 1:
        # Computes and sets the dequantizer recorded variables into the akida layer
        scales = _get_dequantizer_scales(block[0])
        layer_variables = layer_ak.variables
        # Project scales in the final layer activation step
        broadcast_and_set_variable(layer_variables, "act_step",
                                   layer_variables["act_step"] / scales)


def _convert_dequantizer_v2(model_ak, block, inbound_layers_ak):
    """Converts Dequantizer layer and set its variables into the Akida v2 model.

    Args:
        model_ak (:obj:`akida.Model`): the model where the layer will be added.
        block (list(:obj:`keras.Layer`)): the block of keras layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    dequantizer = block[0]

    # Create and add one akida.Dequantizer to the model
    layer_ak = akida.Dequantizer(name=dequantizer.name)
    model_ak.add(layer_ak, inbound_layers_ak)
    # Computes and sets the dequantizer recorded variables into the akida layer
    scales = _get_dequantizer_scales(dequantizer)
    broadcast_and_set_variable(layer_ak.variables, "scales", scales)


def _block_additional_checks(block):
    """Additional checks done in the dequantizer layer block.

    Args:
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
    """
    dequantizer = block[0]
    assert isinstance(dequantizer, Dequantizer)
    # Extract the Dequantizer variables
    frac_bits = dequantizer.frac_bits
    if isinstance(frac_bits, (list, tuple)):
        if len(frac_bits) > 1:
            raise RuntimeError(f"Multi-inbounds in {dequantizer.name} is not supported.")


class DequantizerBlockConverterV1(BlockConverter):
    """Main class that should be used to check if the dequantizer layer block is compatible to an
    Akida V1 conversion and provides a method to convert and integrate its variables in the last
    Akida model layer.

    Args:
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        _block_additional_checks(self._block)

    def convert(self, model_ak, inbounds=[]):
        _convert_dequantizer_v1(model_ak, self._block)


class DequantizerBlockConverterV2(BlockConverter):
    """Main class that should be used to check if the dequantizer layer block is compatible to an
    Akida V2 conversion and provides a method to convert it in a corresponding Akida v2 Dequantizer
    layer.

    Args:
        block (list(:obj:`keras.Layer`)): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        super().__init__(block)
        _block_additional_checks(self._block)

    def convert(self, model_ak, inbounds):
        _convert_dequantizer_v2(model_ak, self._block, inbounds)


# Register the valid dequantizer block pattern for Akida v1 and v2
register_conversion_patterns(AkidaVersion.v1, _PATTERNS, DequantizerBlockConverterV1)
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, DequantizerBlockConverterV2)
