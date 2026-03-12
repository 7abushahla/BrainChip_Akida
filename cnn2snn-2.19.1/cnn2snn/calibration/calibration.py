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
"""Quantized model calibration tools"""

import tf_keras as keras
import tensorflow as tf

from ..quantization_layers import (QuantizedReLU, QuantizedDense,
                                   QuantizedConv2D, QuantizedSeparableConv2D)
from ..quantization_ops import WeightQuantizer

from ..transforms.clone import clone_model_with_weights


def create_submodel(model, first_layer, last_layer):
    """Create a submodel of a Keras Model

    Args:
        model (:obj:`keras.Model`): a Keras Model
        first_layer (:obj:`keras.Layer`): the submodel first layer
        last_layer (:obj:`keras.Layer`): the submodel last layer

    Returns:
        :obj:`keras.Model`: a Keras sub-model
    """
    submodel = keras.Sequential()
    input_shape = first_layer.input_shape[1:]
    submodel.add(keras.layers.Input(shape=input_shape))
    add_layer = False
    for layer in model.layers:
        if layer == first_layer:
            add_layer = True
        if add_layer:
            submodel.add(layer)
        if layer == last_layer:
            break
    return submodel


class NoneQuantizer(WeightQuantizer):
    """A WeightQuantizer that returns float weights
    """

    def scale_factor(self, _):
        return 1

    def quantize(self, w):
        return w


def disable_quantizers(layer):
    """Disable quantizers for the specified layer

    Args:
        layer (:obj:`keras.Layer`): the layer

    Returns:
        [:obj:`cnn2snn.quantization_ops.WeightQuantizer`]: the list of disabled
        quantizers.
    """
    quantizers = [layer.quantizer]
    layer.quantizer = NoneQuantizer(layer.quantizer.bitwidth)
    if hasattr(layer, 'quantizer_dw'):
        quantizers.append(layer.quantizer_dw)
        layer.quantizer_dw = NoneQuantizer(layer.quantizer_dw.bitwidth)
    return quantizers


def restore_quantizers(layer, quantizers):
    """Restores quantizers for the specified layer

    Args:
        layer (:obj:`keras.Layer`): the layer
        quantizers ([:obj:`cnn2snn.quantization_ops.WeightQuantizer`]): the list of disabled
            quantizers.
    """
    layer.quantizer = quantizers[0]
    if hasattr(layer, 'quantizer_dw'):
        layer.quantizer_dw = quantizers[1]


def next_activation(layer):
    """Find the next activation after the specified layer

    Args:
        layer (:obj:`keras.Layer`): the layer

    Returns:
        :obj:`keras.Layer`: the next activation layer or None

    """
    outbound_nodes = layer.outbound_nodes
    for node in outbound_nodes:
        next_layer = node.layer
        if isinstance(next_layer, QuantizedReLU):
            return next_layer
        activation = next_activation(next_layer)
        if activation is not None:
            return activation
    return None


def is_on_top(layer, bottom_layer):
    """Checks if a layer follows another layer

    Args:
        layer (:obj:`keras.Layer`): the top layer
        bottom_layer (:obj:`keras.Layer`): the previous layer

    Returns:
        bool: True if the layer is on top
    """
    outbound_nodes = bottom_layer.outbound_nodes
    if len(outbound_nodes) == 0:
        # Bottom layer has no successors
        return False
    for node in outbound_nodes:
        # Check if layer is on top of one of the outbound layers
        next_layer = node.layer
        if next_layer == layer or is_on_top(layer, next_layer):
            return True
    return False


def is_quantized_neural(layer):
    """Check is a layer is a quantized neural layer

    These layers are the only ones eligible for calibration.

    Args:
        layer (:obj:`keras.Layer`): the top layer

    Returns:
        bool: True if the layer can be calibrated
    """
    return isinstance(
        layer, (QuantizedDense, QuantizedConv2D, QuantizedSeparableConv2D))


class QuantizationSampler():
    """A tool to inspect the layer outputs of a quantized model

    The sampler is initialized with a quantized model and a set of samples used
    for the evaluation of the layer outputs. An optional batch size can be
    specified to avoid out-of-memory errors when using a GPU.

    To evaluate the outputs of a specific layer, it must first be selected
    using the `select_layer` member.

    Once done, three methods are available to inspect the layer outputs:

    - `quantized_outputs` returns the actual outputs of the layer,
    - `float_outputs` returns the outputs of the layer if its weights were not
      quantized,
    - `quantization_error` applies the `keras.metrics.Metric` passed as
      arguments to the difference between the float and quantized outputs.

    Example:

        >>> # Evaluate the quantization MSE of a quantized layer
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(5, input_shape=(3,)),
        ...     keras.layers.ReLU()])
        >>> model_quantized = cnn2snn.quantize(model,
        ...                                    weight_quantization=4,
        ...                                    activ_quantization=4)
        >>> # Instantiate a QuantizationSampler with a few dataset samples
        >>> sampler = QuantizationSampler(model_quantized, samples)
        >>> # Select the quantized layer
        >>> sampler.select_layer(model_quantized.layers[0])
        >>> # Evaluate the Mean Squared Error
        >>> m = keras.metrics.MeanSquaredError()
        >>> mse = sampler.quantization_error(m)

    Args:
        model (:obj:`keras.Model`): a quantized Keras model
        samples (:obj:`np.ndarray`): a set of calibration samples
        batch_size (int): the batch size used for evaluation.
    """

    def __init__(self, model, samples, batch_size=None):
        self.model = model
        n_inputs = samples.shape[0]
        if batch_size is None:
            batch_size = n_inputs
        # We only process full batches
        self.n_batches = n_inputs // batch_size
        # Split samples by batches
        self.samples = []
        for i in range(self.n_batches):
            b_from = i * batch_size
            b_to = (i + 1) * batch_size
            self.samples.append(samples[b_from:b_to, :, :, :])
        self.inputs = None
        self.submodel = None
        self.layer = None

    def select_layer(self, layer, include_activation=False):
        """Select the layer to inspect

        Args:
            layer (:obj:`keras.Layer`): a quantized keras layer
            include_activation (bool): evaluate the outputs after activation
        """
        if not is_quantized_neural(layer):
            raise ValueError("Output layer must be a Quantized layer")
        # Evaluate the outputs immediately before the selected layer to save
        # computation time when repeatedly evaluating the layer outputs
        if self.submodel is None or not is_on_top(layer, self.layer):
            # Evaluate layer inputs from the beginning of the model
            input_layer = self.model.layers[0]
            self.inputs = []
            for i in range(self.n_batches):
                self.inputs.append(self.samples[i])
        else:
            # Evaluate layer inputs using the outputs of the current submodel
            input_layer = self.submodel.layers[-1].outbound_nodes[0].layer
            for i in range(self.n_batches):
                self.inputs[i] = self.quantized_outputs(i)
        if layer != input_layer:
            # Advance inputs
            prev_layer = layer.inbound_nodes[0].inbound_layers
            submodel = create_submodel(self.model, input_layer, prev_layer)
            for i in range(self.n_batches):
                self.inputs[i] = submodel(self.inputs[i])
        # Create the submodel including the layer (and optionally its activation)
        activation = next_activation(layer) if include_activation else None
        if activation is not None:
            self.submodel = create_submodel(self.model, layer, activation)
        else:
            self.submodel = create_submodel(self.model, layer, layer)
        # Remember current layer
        self.layer = layer

    def quantized_outputs(self, batch_index=0):
        """Evaluates the quantized outputs of the selected layer

        Args:
            batch_index (int): the samples batch index

        Returns:
            :obj:`tf.Tensor`: the quantized outputs
        """
        if self.submodel is None:
            raise SystemError("Set output layer before requesting data")
        if batch_index < 0 or batch_index >= self.n_batches:
            raise ValueError("Invalid batch index")
        return self.submodel(self.inputs[batch_index])

    def float_outputs(self, batch_index=0):
        """Evaluates the 'float' outputs of the selected layer

        This disables the weights quantization and evaluates the outputs of the
        layer as if it was using float weights.

        Args:
            batch_index (int): the samples batch index

        Returns:
            :obj:`tf.Tensor`: the quantized outputs
        """
        if self.submodel is None:
            raise SystemError("Set output layer before requesting data")
        if batch_index < 0 or batch_index >= self.n_batches:
            raise ValueError("Invalid batch index")
        quantizers = disable_quantizers(self.layer)
        outputs = self.submodel(self.inputs[batch_index])
        restore_quantizers(self.layer, quantizers)
        return outputs

    def quantization_error(self, metric):
        """Evaluates the quantization error of the selected layer

        This evaluates the difference between the layer quantized outputs and
        the outputs if the layer was using float weights using the Metric
        passed as parameter.

        Args:
            metric (:obj:`keras.metrics.Metric`): the Metric for evaluation

        Returns:
            the Metric results (usually a single float)

        """
        metric.reset_state()
        for i in range(self.n_batches):
            metric.update_state(self.float_outputs(i),
                                self.quantized_outputs(i))
        return metric.result().numpy()


def previous_neural(layer):
    """Find the previous neural layer before the specified layer

    Args:
        layer (:obj:`keras.Layer`): the layer

    Returns:
        :obj:`keras.Layer`: the next activation layer or None

    """
    inbound_nodes = layer.inbound_nodes
    for node in inbound_nodes:
        previous_layer = node.inbound_layers
        if is_quantized_neural(previous_layer):
            return previous_layer
        neural_layer = previous_neural(previous_layer)
        if neural_layer is not None:
            return neural_layer
    return None


def activations_rescaling(model, samples, batch_size=None):
    """iterate over the layers of a quantized model, evaluating
    the pre-activations before each QuantizedReLu layer,
    and adjusting the max_value parameter of each QuantizedReLu
    accordingly.

    Args:
        model (:obj:`keras.Model`): a Keras Model
        samples (:obj:`np.ndarray`): a set of calibration samples
        batch_size (int): the batch size used for evaluation.

    """
    clone_model = clone_model_with_weights(model)
    sampler = QuantizationSampler(clone_model, samples, batch_size)
    for layer in clone_model.layers:
        if not isinstance(layer, QuantizedReLU):
            continue
        previous_neural_layer = previous_neural(layer)
        sampler.select_layer(previous_neural_layer, True)
        float_output = sampler.float_outputs()
        max_val = tf.reduce_max(float_output)
        if max_val < layer.max_value_:
            layer.max_value_ = tf.constant(max_val, dtype=tf.float32)

    return clone_model
