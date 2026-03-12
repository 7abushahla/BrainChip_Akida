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
"""Functions to convert input layers to Akida layers.
"""
import tensorflow as tf

from akida import Model, InputData
from ..akida_versions import AkidaVersion, get_akida_version


def get_tensor_dtype(keras_tensor, signed=True):
    """Returns the dtype of ``keras_tensor``

    Args:
        keras_tensor (:obj:`KerasTensor`): the Keras tensor to read the dtype from.
        signed (bool, optional): if a floating type is found, returns tf.int8 if signed=True,
            otherwise tf.uint8. Defaults to True.

    Returns:
        tf.Dtype: The input dtype.
    """
    input_dtype = keras_tensor.dtype
    if not input_dtype.is_integer:
        # Returns a default dtype following signed, in order to validate
        # quantizations made prior to QML 0.13.0
        return tf.int8 if signed else tf.uint8
    return input_dtype


def _create_input_data(keras_tensor):
    """Parses a KerasTensor and returns the corresponding Akida InputData layer.

    Args:
        keras_tensor (:obj:`KerasTensor`): the Keras tensor to convert.

    Returns:
        akida.InputData: The parsed layer.
    """
    input_shape = keras_tensor.shape
    input_dtype = get_tensor_dtype(keras_tensor)
    if input_dtype.is_unsigned:
        raise ValueError("Only signed inputs are supported.")
    # Ignore first dimension, which is batch size
    input_shape = input_shape[1:]
    # input shape must not exceeds 3 dim
    if len(input_shape) > 3:
        raise ValueError("input shape must not exceed 3 dimensions."
                         f"Receives {input_shape}")

    # Create a list with 1 for each dimension that we miss before having 3
    # dims, that is what akida expects
    missing_dimensions = [1] * (3 - len(input_shape))
    input_shape = [*missing_dimensions, *input_shape]
    # the akida version determines the input_bits
    if get_akida_version() == AkidaVersion.v1:
        input_bits = 4
    else:
        input_bits = input_dtype.size * 8
    # With the convert api of cnn2snn the model should handle only unsigned inputs.
    return InputData(input_shape=input_shape, input_bits=input_bits, name=keras_tensor.name)


def _create_input_data_from_layer(layer):
    """Creates an Akida InputData layer from a previous akida layer.

    Args:
        layer (:obj:`akida.Layer`): the inbound of the InputData.

    Returns:
        akida.InputData: the parser layer.
    """
    input_shape = layer.output_dims
    input_bits = layer.parameters.output_bits
    return InputData(input_shape=input_shape, input_bits=input_bits)


def convert_input(model_ak, model_k, inbounds=[]):
    """Converts a keras Input layer into ``akida.InputData`` and add it to the
    Akida's model.

    Args:
        model_ak (:obj:`akida.Model`): the Akida model where the Akida InputData will be inserted.
        model_k (:obj:`keras.Layer`): the keras quantized model.
        inbounds (list, optional): list of inbound Akida layers. Defaults to [].

    Returns:
        akida.Layer: the created Akida InputData layer.
    """
    if not isinstance(model_ak, Model):
        raise TypeError(f"Expecting an akida model, received {type(model_ak)}")

    # Get InputData from model input or inbounds.
    if len(inbounds) > 0:
        input_data = _create_input_data_from_layer(inbounds[0])
    else:
        input_data = _create_input_data(model_k.input)
    model_ak.add(input_data, inbounds)
    return input_data
