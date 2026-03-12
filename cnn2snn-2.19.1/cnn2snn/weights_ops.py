#!/usr/bin/env python
# ******************************************************************************
# Copyright 2019 Brainchip Holdings Ltd.
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
"""Functions to convert weights from Keras to Akida.

"""
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tf_keras.layers as layers
from akida import LayerType

KerasLayerParams = namedtuple(
    'KerasLayerParams',
    ['bias', 'act_threshold', 'act_step', 'global_avg_pool_factor'])


def set_layer_variables(layer_ak,
                        input_scaling,
                        layer_neural,
                        layer_activation=None,
                        layer_globalavgpool=None):
    """Computes and sets weights/thresholds for an Akida layer.

    This function converts the weights and variables of a sequence of Keras
    layers (a neural layer, an optional global average pooling layer and an
    optional activation layer) into the weights and thresholds for the given
    Akida layer.

    The input scaling corresponds either to the input scaling of the model if
    the Akida layer is the first layer, or the input scaling of the previous
    activation layer if the Akida layer is an intermediate or final layer.
    """

    # Convert weights
    weights, weights_scale_factor = \
        convert_layer_weights(layer_ak.parameters.layer_type, layer_neural,
                              layer_ak.input_dims)

    # Convert fire thresholds and steps
    th_fire, th_step = _convert_layer_thresholds(layer_ak.parameters.layer_type,
                                                 layer_neural, layer_activation,
                                                 layer_globalavgpool,
                                                 input_scaling,
                                                 weights_scale_factor, weights)

    # Set weights
    if layer_ak.parameters.layer_type == LayerType.SeparableConvolutional:
        layer_ak.set_variable('weights', weights[0])
        layer_ak.set_variable('weights_pw', weights[1])
    else:
        layer_ak.set_variable('weights', weights)

    # Set thresholds
    layer_ak.set_variable('threshold', th_fire)
    layer_ak.set_variable("act_step", th_step)


def convert_layer_weights(layer_type, layer_k, input_dims=None):
    """Converts weights for an Akida layer.

    This function returns the converted weights for any Akida layer,
    according to the Akida layer type and the Keras neural layer.

    Args:
        layer_type (str): type of the layer Akida.
        layer_k (:obj:`keras.Layer`): the Keras layer from which the weights
            will be converted.
        input_dims (tuple): input dimensions of the Keras layer. This argument
            is only required for a fullyConnected Akida layer.

    Returns:
        tuple: the converted weights and the weights scale factor from
            quantization.

    """
    if layer_type == LayerType.InputConvolutional:
        return _convert_conv_weights(layer_k, input_conv=True)
    if layer_type == LayerType.Convolutional:
        return _convert_conv_weights(layer_k)
    if layer_type == LayerType.SeparableConvolutional:
        return _convert_separable_conv_weights(layer_k)
    if layer_type == LayerType.FullyConnected:
        return _convert_dense_weights(layer_k, input_dims)
    assert False, "The layer type is unmanaged."
    return tuple()


def _convert_conv_weights(layer, input_conv=False):
    # Quantize weights
    weights = layer.get_weights()[0]
    wq, delta = quantize_weights(layer.quantizer, weights)

    wq_akida = wq * delta
    wq_akida = np.round(wq_akida).astype(np.int8)

    # Flip W and H dimensions for conv. kernels (not input conv.)
    if not input_conv:
        wq_akida = np.flip(wq_akida, axis=[0, 1])

    return wq_akida, delta


def _convert_separable_conv_weights(layer):
    # Quantize depthwise weights
    weights = layer.get_weights()[0]
    wq_dw, delta_dw = quantize_weights(layer.quantizer_dw, weights)
    w_ak_dw = wq_dw * delta_dw
    w_ak_dw = np.round(w_ak_dw).astype(np.int8)
    w_ak_dw = np.flip(w_ak_dw, axis=[0, 1])

    # Quantize pointwise weights
    weights_pw = layer.get_weights()[1]
    wq_pw, delta_pw = quantize_weights(layer.quantizer, weights_pw)
    # Pointwise weights in Keras have HWCN format and H = W = 1. This
    # makes the conversion to Akida's NCHW trivial.
    w_ak_pw = wq_pw * delta_pw
    w_ak_pw = np.round(w_ak_pw).astype(np.int8)

    # For separable, the weights scale factor can be seen as the product of the
    # DW and PW scale factors.
    delta = delta_dw * delta_pw
    return (w_ak_dw, w_ak_pw), delta


def _convert_dense_weights(layer, input_dims):
    # Quantize weights
    weights = layer.get_weights()[0]
    wq, delta = quantize_weights(layer.quantizer, weights)

    # retrieve input dimensions from Akida's layer
    inX, inY, inchans = input_dims
    # When a fully connected layer follows a convolutional layer, we need to modify
    # the way the weights are laid out because the Keras and Akida flatten operations
    # that happen on spatial dimensions are inverted (row-major versus col-major).
    # We therefore need to:
    # - reshape the Keras (FxN) weights to match the input shape (XxYxCxN),
    # - transpose to the equivalent akida shape (CxYxXxN),
    # - reshape back to a flat shape (1x1xFxN).
    #
    wq_akida = wq.reshape(inX, inY, inchans, layer.units) \
        .transpose(2, 1, 0, 3) \
        .reshape(1, 1, inchans * inY * inX, layer.units)
    # Multiply by delta, round and cast to int
    wq_akida = wq_akida * delta
    wq_akida = np.round(wq_akida).astype(np.int8)

    return wq_akida, delta


def _prepare_threshold_calculations(layer_ak_type,
                                    layer_neural,
                                    layer_activation=None,
                                    layer_globalavgpool=None):
    """Retrieves required variables to calculate Akida fire threshold and
    step.

    This function returns a dictionary with variables required for threshold
    calculations of the given Akida layer map: bias in the Keras neural layer,
    parameters of the activation layer and global average pooling factor.

    Args:
        layer_ak_type (:obj:`akida.LayerType`): the type of the Akida layer.
        layer_neural (:obj:`keras.Layer`): the Keras neural layer.
        layer_activation (:obj:`keras.Layer`): the Keras activation layer.
        layer_globalavgpool (:obj:`keras.Layer`): the optional Keras global
            average pooling layer.

    Returns:
        dict: required variables to calculate Akida fire threshold and step.
    """

    # Initialize variables to their default values
    threshold_shape = layer_neural.output_shape[-1]
    bias = np.zeros(threshold_shape)
    thresh_keras = np.zeros(threshold_shape)
    step_keras = np.ones(threshold_shape)
    global_avg_pool_factor = 1

    # Get bias if present
    if layer_neural.use_bias:
        if layer_ak_type == LayerType.SeparableConvolutional:
            bias = layer_neural.get_weights()[2]
        else:
            bias = layer_neural.get_weights()[1]

    # Get parameters from the discrete ReLU activation (threshold and step)
    if layer_activation:
        thresh_keras = (layer_activation.threshold.numpy() * np.ones(threshold_shape))
        step_keras = (
            (layer_activation.step.numpy() *
             2.**layer_activation.bitwidth /
             16) *
            np.ones(threshold_shape))

    # Get global average pooling factor
    if isinstance(layer_globalavgpool, layers.GlobalAveragePooling2D):
        global_avg_pool_factor = np.prod(layer_globalavgpool.input_shape[1:3])

    return KerasLayerParams(bias, thresh_keras, step_keras,
                            global_avg_pool_factor)


def _convert_layer_thresholds(layer_ak_type, layer_neural, layer_activation,
                              layer_globalavgpool, input_scaling,
                              weights_scale_factor, weights_ak):
    """Returns the fire thresholds and threshold steps for a given Akida layer.

    This function computes the fire thresholds and steps for a given Akida layer
    from the corresponding neural, activation and global avg pooling Keras
    layers. Other arguments are required to compute these variables, such as
    the input scaling, the weights scale factor, and the Akida weights (only
    used for an InputConvolutional Akida layer).

    Args:
        layer_ak_type (:obj:`akida.LayerType`): the type of the Akida layer.
        layer_neural (:obj:`keras.Layer`): the Keras neural layer.
        layer_activation (:obj:`keras.Layer`): the Keras activation layer.
        layer_globalavgpool (:obj:`keras.Layer`): the optional Keras global
            average pooling layer.
        input_scaling (2-element tuple): the input scaling of the current layer.
        weights_scale_factor (float): the weights scale factor of the current
            Akida layer. This is given by the 'convert_layer_weights' function.
        weights_ak (np.ndarray): the Akida weights of the current Akida layer.

    Returns:
        tuple: the fire thresholds and steps.

    """

    input_scale, input_shift = input_scaling
    p = _prepare_threshold_calculations(layer_ak_type, layer_neural,
                                        layer_activation, layer_globalavgpool)

    # Compute threshold fires and steps
    th_fire = (input_scale * weights_scale_factor * (p.act_threshold - p.bias))
    th_step = (input_scale * weights_scale_factor * p.act_step).astype(
        np.float32)

    # If there is an InputConv layer, update th_fire with input shift
    if layer_ak_type == LayerType.InputConvolutional:
        th_fire += np.sum(weights_ak, axis=(0, 1, 2)) * input_shift

    th_fire *= p.global_avg_pool_factor
    th_step *= p.global_avg_pool_factor

    th_fire = np.floor(th_fire).astype(np.int32)

    return th_fire, th_step


def quantize_weights(quantizer, w):
    """Returns quantized weights and delta as numpy arrays.

    Internally, it uses a tf.function that wraps calls to the quantizer in
    a graph, allowing the weights to be quantized eagerly.

    Args:
        quantizer (:obj:`WeightQuantizer`): the quantizer object.
        w (:obj:`np.ndarray`): the weights to quantize.

    Returns:
        :obj:`np.ndarray`: the quantized weights `np.ndarray` and the scale
            factor scalar.

    """

    w_tf = tf.constant(w)
    wq = quantizer.quantize(w_tf)
    scale_factor = quantizer.scale_factor(w_tf)
    return wq.numpy(), scale_factor.numpy()
