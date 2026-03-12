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
"""Functions to convert QuantizedConcatenate to Akida.
"""
from akida import Concatenate
from quantizeml.layers import QuantizedConcatenate

from .block_converter import BlockConverter, register_conversion_patterns
from ..akida_versions import AkidaVersion

__all__ = ["ConcatenateBlockConverter"]

_PATTERNS = [(QuantizedConcatenate,)]


def convert_quantized_concatenate(model_ak, block, inbound_layers_ak):
    """Converts QuantizedConcatenate layer and its variables and adds it to the
    Akida's model.

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the model will be added.
        block (list(:obj:`keras.Layer`)): the block of keras layers.
        inbound_layers_ak (list(:obj:`akida.Layer`)): list of inbound Akida layers.
    """
    concat = block[0]
    # Create and add layer to the akida model
    # It is quite simple because there are no variables or params to be set.
    layer_ak = Concatenate(name=concat.name)
    model_ak.add(layer_ak, inbound_layers_ak)


class ConcatenateBlockConverter(BlockConverter):
    """Main class that should be used to check if the concatenate layer block is compatible to an
    Akida v2 conversion and provides a method to convert it in an equivalent Akida Concatenate
    layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def convert(self, model_ak, inbounds):
        convert_quantized_concatenate(model_ak, self._block, inbounds)


# Register the valid concatenate block pattern for Akida v2
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, ConcatenateBlockConverter)
