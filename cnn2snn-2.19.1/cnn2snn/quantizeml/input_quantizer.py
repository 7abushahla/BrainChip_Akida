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
"""Functions to convert an InputQuantizer into an Akida Quantizer layer.
"""

__all__ = ["InputQuantizerBlockConverter"]

import akida
from quantizeml.layers import InputQuantizer

from ..akida_versions import AkidaVersion
from .block_converter import BlockConverter, register_conversion_patterns
from .weights import broadcast_and_set_variable


_PATTERNS = [(InputQuantizer,)]


class InputQuantizerBlockConverter(BlockConverter):
    """Main class that should be used to convert an InputQuantizer in an
    equivalent Akida Quantizer layer.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def convert(self, model_ak, inbounds):
        input_quantizer = self._block[0]
        if (N := len(input_shape := tuple(input_quantizer.input_shape[1:]))) < 3:
            # Fill the missing dimensions with ones.
            input_shape = (1,) * (3 - N) + input_shape

        # Create the Akida Quantizer layer from the InputQuantizer layer.
        ak_quantizer = akida.Quantizer(
            input_shape=input_shape,
            output_bits=input_quantizer.bitwidth,
            output_signed=input_quantizer.signed,
            name=input_quantizer.name
        )

        # Add Quantizer into the Akida model.
        model_ak.add(ak_quantizer, inbounds)

        # Set layer variables.
        scales = 2.0 ** input_quantizer.frac_bits.value.numpy()
        broadcast_and_set_variable(ak_quantizer.variables, "scales", scales)
        if zero_points_recorder := getattr(input_quantizer, "zero_points", None):
            zero_points = zero_points_recorder.value.values.numpy().astype("uint8")
            broadcast_and_set_variable(ak_quantizer.variables, "zero_points", zero_points)


# Register the valid input conv block pattern for Akida v1 and v2
register_conversion_patterns(AkidaVersion.v1, _PATTERNS, InputQuantizerBlockConverter, True)
register_conversion_patterns(AkidaVersion.v2, _PATTERNS, InputQuantizerBlockConverter, True)
