# ******************************************************************************
# Copyright 2020 Brainchip Holdings Ltd.
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
"""Implementation of the Bias Correction algorithm from:
   Data-Free Quantization Through Weight Equalization and Bias Correction
   Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling
   https://arxiv.org/abs/1906.04721
"""
import tf_keras as keras
import numpy as np
import tensorflow as tf

from .calibration import QuantizationSampler, is_quantized_neural


def get_bias_index(layer):
    """Get the index of the bias weights

    Args:
        layer (:obj:`keras.Layer`): a Keras neural layer.

    Returns:
        int: the index of the bias in the weights list
    """
    if isinstance(layer, keras.layers.SeparableConv2D):
        return 2
    if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
        return 1
    raise ValueError("{} is not a neural layer".format(layer.name))


def clone_model_with_biases(model):
    """Clones the model and add zero biases if needed

    The cloned model is identical to the input model, except that zero biases
    are added to neural layers that did not use them.

    Args:
        model (:obj:`keras.Model`): a Sequential Keras model.

    Returns:
        :obj:`keras.Model`: a Sequential Keras model.
    """

    assert isinstance(model, keras.Sequential)

    clone_model = keras.Sequential()
    clone_model.add(keras.layers.Input(model.input_shape[1:]))
    for layer in model.layers:
        config = layer.get_config()
        if is_quantized_neural(layer) and not config['use_bias']:
            # Modify configuration to use a bias
            config['use_bias'] = True
            # Create a cloned layer
            clone_layer = layer.__class__.from_config(config)
            clone_model.add(clone_layer)
            # Get original weights
            w_list = layer.get_weights()
            # Insert zero bias
            bias_index = get_bias_index(layer)
            bias_shape = clone_layer.get_weights()[bias_index].shape
            w_list.insert(bias_index, np.zeros(bias_shape, dtype=np.float32))
            # Update cloned layer weights
            clone_layer.set_weights(w_list)
        else:
            # Simply clone the layer
            clone_layer = layer.__class__.from_config(config)
            clone_model.add(clone_layer)
            clone_layer.set_weights(layer.get_weights())
    return clone_model


def bias_correction(model, samples, batch_size=None):
    """Apply a corrective bias to quantized layers.

    This implements the Bias Correction algorithm described in:
    Data-Free Quantization Through Weight Equalization and Bias Correction
    Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling
    https://arxiv.org/abs/1906.04721

    It is empirically demonstrated in the original paper that the weight
    quantization can introduce a biased error in the activations that is quite
    significant for low bitwidth weights (i.e. lower than 8-bit).
    This algorithm simply estimates the quantization bias on a set of samples,
    and subtracts it from the layer bias variable.

    If the accuracy of the quantized model suffers a huge drop as compared to
    the original model, this simple correction can recover the largest part of
    the drop, but not all of it.

    When optimizing a model, nothing is required but a set of samples for
    calibration (typically from the training dataset).
    Depending on the model and dataset, your mileage may vary, but it has been
    observed empirically that there is no significant difference between the
    models fixed with a very few samples (16) and those fixed with a higher
    number of samples (1024).

    Args:
        model (:obj:`keras.Model`): a quantized Keras Model
        samples (:obj:`np.ndarray`): a set of samples used for calibration
        batch_size (int): the batch size used when evaluating samples

    Returns:
        keras.Model: a quantized Keras model whose biases have been
        corrected

    """
    # Clone the model, adding biases to layers that don't use them
    bc_model = clone_model_with_biases(model)
    m = keras.metrics.MeanSquaredError()
    sampler = QuantizationSampler(bc_model, samples, batch_size)
    # Adjust bias of each layer iteratively
    for layer in bc_model.layers:
        if is_quantized_neural(layer):
            print("Adjusting bias for {}".format(layer.name))
            sampler.select_layer(layer, include_activation=False)
            # Evaluate quantization error
            err_before = sampler.quantization_error(m)
            # Iterate over sample batches to evaluate raw error
            q_bias = 0
            n_batches = sampler.n_batches
            float_outputs = []
            for i in range(n_batches):
                # Evaluate the layer outputs
                outputs = sampler.quantized_outputs(i)
                # Evaluate and store the layer float outputs
                float_outputs.append(sampler.float_outputs(i))
                # Evaluate quantization error
                error = float_outputs[i] - outputs
                axis = tf.range(tf.rank(error) - 1)
                # Add the contribution of this batch to the corrective bias
                q_bias += tf.math.reduce_mean(error, axis=axis) / n_batches
            # Adjust quantized layer bias
            w_list = layer.get_weights()
            w_list[get_bias_index(layer)] += q_bias
            layer.set_weights(w_list)
            # Evaluate quantization error compared to previous float outputs
            m.reset_state()
            for i in range(n_batches):
                # Evaluate the layer outputs
                outputs = sampler.quantized_outputs(i)
                m.update_state(float_outputs[i], outputs)
            err_after = m.result().numpy()
            print(f"quantization error: {err_before:.4f} -> {err_after:.4f}")
    return bc_model
