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
"""
Tools to facilitate quantizeml models conversion to Akida pipelines.
"""

from ..akida_versions import AkidaVersion

# Mapper to match layers block patterns with the right conversion method, populated thanks to
# `register_conversion_patterns` function.
_V2_PATTERN_CONVERTERS = {}
_V1_PATTERN_CONVERTERS = {}
_V2_INPUT_PATTERN_CONVERTERS = {}
_V1_INPUT_PATTERN_CONVERTERS = {}


def register_conversion_patterns(version, patterns, converter, input_pattern=False):
    """Helper to populate the patterns_to_function dictionaries according to the right version.

    Args:
        version (AkidaVersion): the target Akida version.
        patterns (list): list of compatible quantized layers type patterns.
        converter (:obj:`BlockConverter`): the associated converter object.
        input_pattern (bool, optional): boolean to specify if the pattern should match an Akida
            input layer (such as InputConv2D). Defaults to False.
    """
    if input_pattern:
        if version == AkidaVersion.v2:
            dict_ver = _V2_INPUT_PATTERN_CONVERTERS
        else:
            dict_ver = _V1_INPUT_PATTERN_CONVERTERS
    else:
        dict_ver = _V2_PATTERN_CONVERTERS if version == AkidaVersion.v2 else _V1_PATTERN_CONVERTERS
    for pattern in patterns:
        dict_ver.update({pattern: converter})


class BlockConverter:
    """This represents the main class that allows the conversion of a block of quantized layers with
    quantizeml into their Akida layer equivalent.

    It extracts the layers block pattern, checks if it's valid, applies optional additional checks
    (that should be implemented and integrated in the sub class constructor) and will allow the
    conversion of the block through the method convert.

    Args:
        block (list): list of quantizeml quantized layers.
    """

    def __init__(self, block):
        self._block = block

    def convert(self, model_ak, inbounds):
        raise NotImplementedError

    @property
    def pattern(self):
        return [layer.__class__ for layer in self._block]

    @property
    def block_names(self):
        return [layer.name for layer in self._block]

    @property
    def root(self):
        return self._block[0]
