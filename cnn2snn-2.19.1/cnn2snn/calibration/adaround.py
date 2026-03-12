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
"""Implementation of the Adaround algorithm described in:
   Up or Down? Adaptive Rounding for Post-Training Quantization
   Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen
   Blankevoort
   https://arxiv.org/abs/2004.10568
"""
import shutil
from math import pi
import tf_keras as keras
import numpy as np
import tensorflow as tf

from ..transforms.clone import clone_model_with_weights
from ..quantization_layers import QuantizedDense, QuantizedConv2D, QuantizedSeparableConv2D
from ..quantization_ops import WeightQuantizer
from .calibration import QuantizationSampler, is_quantized_neural

# Rectified sigmoid constants
GAMMA = -0.1
ZETA = 1.1
# Rounding loss annealing constants
BETA_START = 20
BETA_END = 2
# Rounding loss regularization
RD_REG = 0.01


class AdaRoundQuantizer(WeightQuantizer):
    """Adaptive Rounding weights quantizer

    Replaces original weights by floored weights + decimal.

    It relies on an internal continuous variable to evaluate the decimals of
    each weights.

    It provides a rounding loss method that allows an optimizer to force the
    decimals to take the values 0 or 1.

    It also implements the WeightQuantizer API to be able to replace the
    original quantizers of a Model, and return the reconstructed weights
    instead of the quantized weights.

    Args:
        quantizer (:obj:`cnn2snn.quantization_ops.WeightQuantizer`): the
           original quantizer.
        weights (:obj:`np.ndarray`): the original weights
    """

    def __init__(self, quantizer, weights, **kwargs):
        # Store original quantizer
        self.quantizer = quantizer
        # Cache scale factor for the original weights
        self.sf = quantizer.scale_factor(weights)
        # Evaluate rescaled weights
        w_scaled = tf.constant(weights) * self.sf
        # Store floored rescaled weights
        self.w_floor = tf.constant(tf.math.floor(w_scaled))
        # Evaluate decimals
        d = w_scaled - self.w_floor
        # Create alpha variable, init so that self.decimals() == d
        initial_alpha = -1 * tf.math.log((ZETA - GAMMA) / (d - GAMMA) - 1)
        self.alpha = tf.Variable(name="alpha",
                                 initial_value=initial_alpha,
                                 dtype=tf.float32,
                                 trainable=True)
        # The quantizer behaviour is different at training or evaluation time
        self.training = True
        super().__init__(quantizer.bitwidth, **kwargs)

    def decimals(self):
        """The projection of the alpha variable to [0, 1]

        Returns:
            :obj:`tf.Tensor` : the reconstructed decimals
        """
        # Evaluate sigmoid(alpha) to evaluate the rounding between [0, 1]
        rounding = tf.math.sigmoid(self.alpha)
        # Apply an affine transformation to project to [GAMMA, ZETA]
        rounding = rounding * (ZETA - GAMMA) + GAMMA
        # Clip and return
        return tf.clip_by_value(rounding, 0, 1)

    def loss(self, beta):
        """The rounding loss used to force the decimals to 0 or 1.

        Args:
            beta (int): the loss annealing parameter

        Returns:
            :obj:`tf.Tensor`: the rounding loss Tensor
        """
        if beta == 0:
            return 0
        # The loss is the sum of the distances to the [0, 1] boundaries powered
        # by the beta annealing parameter
        return tf.reduce_sum(1 - (tf.abs(2 * self.decimals() - 1))**beta)

    def status(self):
        """The current rounding status

        This primarily evaluates the number of roamers, i.e. weights that have
        not be rounded yet.
        This also returns the number of floored and ceiled weights.

        Returns:
            (int, int, int): the number of roamers, floored and ceiled weights
        """
        d = self.decimals().numpy()
        num_weights = d.size
        roamers = num_weights - np.sum(np.isin(d, [0, 1]))
        floor_r = np.sum(d <= 0.5) / num_weights
        ceil_r = np.sum(d > 0.5) / num_weights
        return roamers, floor_r, ceil_r

    def recons_weights(self):
        """Reconstruct weights from original weights using current decimals.

        Returns:
            :obj:`tensorflow.Tensor`: a Tensor of float weights.
        """
        # Reconstruct rescaled weights: floored rescaled weights + decimals
        adaround_w = self.w_floor + self.decimals()
        # Clip and rescale to obtain the rounded weights
        return tf.clip_by_value(adaround_w, -self.kmax_, self.kmax_) / self.sf

    def scale_factor(self, _w):
        """Return the original quantizer scale factor

        Args:
          _w (:obj:`tensorflow.Tensor`): ignored.

        Returns:
          :obj:`tensorflow.Tensor`: a Tensor containing a list of scalar values
                (1 or more).
        """
        return self.sf

    def quantize(self, _w):
        """Return layer weights

        Although it takes the original weights as parameters, it ignores it and
        uses instead the reconstructed weights.

        This method has a different behaviour at evaluation and training time.

        - at training time it simply returns the non-quantized reconstructed
          weights to provide a smooth quantization loss
        - at evaluation time it applies the original quantization to the
          reconstructed weights.

        Note that when all decimals have converged to either zero or one, the
        reconstructed weights are aligned on quantization intervals, so the
        returned weights are equivalent to the reconstructed weights.

        Args:
            _w (:obj:`tensorflow.Tensor`): ignored.

        Returns:
            :obj:`tensorflow.Tensor`: a Tensor of quantized weights.
        """
        if self.training:
            return self.recons_weights()
        return self.quantizer.quantize(self.recons_weights())


def initialize_training(optimizer, model):
    """Prepare target model for Adaround optimization

    The original quantizers are replaced by AdaRound quantizers

    Args:
        optimizer (:obj:`tensorflow.keras.optimizers.Optimizer`): an optimizer
        model (:obj:`keras.models.Model`): a quantized Keras Model

    Returns:
        dict of [`AdaRoundQuantizer`], dict of [`tf.Variable`]: a list of AdaRound quantizers and
        a list of trainable variables, grouped by layer

    """
    adarounds = {}
    trainable_variables = {}
    for layer in model.layers:
        if not is_quantized_neural(layer):
            continue
        # Replace standard quantizers by Adaround quantizers
        adarounds[layer] = []
        trainable_variables[layer] = []
        if isinstance(layer, QuantizedSeparableConv2D):
            w_list = layer.get_weights()
            dw = w_list[0]
            layer.quantizer_dw = AdaRoundQuantizer(layer.quantizer_dw, dw)
            adarounds[layer].append(layer.quantizer_dw)
            trainable_variables[layer].append(layer.quantizer_dw.alpha)
            pw = w_list[1]
            layer.quantizer = AdaRoundQuantizer(layer.quantizer, pw)
            adarounds[layer].append(layer.quantizer)
            trainable_variables[layer].append(layer.quantizer.alpha)
        elif isinstance(layer, (QuantizedDense, QuantizedConv2D)):
            w_list = layer.get_weights()
            w = w_list[0]
            layer.quantizer = AdaRoundQuantizer(layer.quantizer, w)
            adarounds[layer].append(layer.quantizer)
            trainable_variables[layer].append(layer.quantizer.alpha)
    # Create all optimizer variables
    all_trainable_variables = sum(trainable_variables.values(), [])
    # Since TensorFlow 2.12, the optimizer must build the variables before use them.
    if hasattr(optimizer, 'build'):
        optimizer.build(all_trainable_variables)
    return adarounds, trainable_variables


def finalize_training(layer):
    """Restore target layer after Adaround optimization

    The original quantizers are restored and the rounded weights are frozen.

    Args:
        layer (:obj:`keras.layers.Layer`): the layer to optimize

    """
    # Freeze rounded weights and restore original quantizers
    if not is_quantized_neural(layer) or not isinstance(layer.quantizer,
                                                        AdaRoundQuantizer):
        return
    if isinstance(layer, QuantizedSeparableConv2D):
        # Get new rounded weights
        adaround_dw = layer.quantizer_dw.recons_weights()
        adaround_pw = layer.quantizer.recons_weights()
        # Restore original quantizers
        layer.quantizer_dw = layer.quantizer_dw.quantizer
        layer.quantizer = layer.quantizer.quantizer
        # Compare quantized weights: original vs rounded
        w_list = layer.get_weights()
        print_console(
            "Changed dw weights: {} / {}".format(
                np.sum(
                    layer.quantizer_dw.quantize(
                        w_list[0]) != layer.quantizer_dw.quantize(adaround_dw)),
                adaround_dw.numpy().size))
        print_console("Changed pw weights: {} / {}".format(
            np.sum(
                layer.quantizer.quantize(w_list[1]) != layer.quantizer.quantize(
                    adaround_pw)),
            adaround_pw.numpy().size))
        # Replace original weights by rounded weights
        w_list[0] = adaround_dw
        w_list[1] = adaround_pw
        layer.set_weights(w_list)
    elif isinstance(layer, (QuantizedConv2D, QuantizedDense)):
        # Get new rounded weights
        adaround_w = layer.quantizer.recons_weights()
        # Restore original quantizer
        layer.quantizer = layer.quantizer.quantizer
        # Compare quantized weights: original vs rounded
        w_list = layer.get_weights()
        print("Changed weights: {} / {}".format(
            np.sum(
                layer.quantizer.quantize(w_list[0]) != layer.quantizer.quantize(
                    adaround_w)),
            adaround_w.numpy().size))
        # Replace original weights by rounded weights
        w_list[0] = adaround_w
        layer.set_weights(w_list)


def compute_beta(step, max_step):
    """Computes the rounding loss annealing parameter

    Args:
        step (int): the current optimization step
        max_step (int): the maximum optimization step

    Returns:
        int: the annealing parameter
    """
    rel_step = step / max_step
    cosine_decay = 0.5 * (1 + tf.math.cos(rel_step * pi))
    return BETA_END + (BETA_START - BETA_END) * cosine_decay


def print_console(message, new_line=True):
    """Print a message on the console.

    If new_line is True, this is equivalent to a simple print.

    If new_line is False, the message is padded with spaces until the end of
    the line, but no end-of-line is inserted, which means that subsequent calls
    to print will overwrite the last line of console.

    Args:
        message (str): the message to print on the console
        new_line (bool): go to the next line after the message has been printed
    """
    if new_line:
        print(message)
    else:
        col, _ = shutil.get_terminal_size()
        print(f"{message:<{col}}", sep='', end='\r', flush=True)


def optimize_rounding(model, samples, layer, optimizer, epochs, loss,
                      batch_size, include_activation, adarounds, trainable_variables):
    """AdaRound main optimization method

    Optimize the AdaRound alpha variables using gradient descent to minimize
    the quantization error.

    Args:
        model (:obj:`keras.models.Model`): a quantized Keras Model
        samples (:obj:`np.ndarray`): a set of samples used for calibration
        layer (:obj:`keras.layers.Layer`): the layer to optimize
        optimizer (:obj:`tensorflow.keras.optimizers.Optimizer`): an optimizer
        epochs (int): the maximum number of epochs
        loss (:obj:`tensorflow.keras.losses.Loss`): the error loss function
        batch_size (int): the batch size used when evaluating samples
        include_activation (bool): quantization error is evaluated after
          activation.
    """
    # Instantiate a sampler and select the target layer
    sampler = QuantizationSampler(model, samples, batch_size)
    sampler.select_layer(layer, include_activation)
    m = keras.metrics.MeanSquaredError()
    # Evaluate quantization error
    err_before = sampler.quantization_error(m)
    float_outputs = None
    # Training loop
    print_console("Optimizing {}:".format(layer.name))
    for e in range(epochs):
        # Evaluate rounding loss parameter
        beta = compute_beta(e, epochs)
        for i in range(sampler.n_batches):

            # If we have a single batch, we can cache the float outputs
            if sampler.n_batches > 1 or float_outputs is None:
                # Evaluate the float outputs for this batch
                float_outputs = sampler.float_outputs(i)

            # Optimize the adaround alpha variables
            with tf.GradientTape() as tape:

                # Compute layer outputs with adaround
                outputs = sampler.quantized_outputs(i)

                # Evaluate quantization loss
                q_loss = loss(float_outputs, outputs)

                # Evaluate rounding loss for each set of weights
                r_loss = 0
                for adr in adarounds:
                    r_loss += RD_REG * adr.loss(beta)

                total_loss = q_loss + r_loss

            # Get gradients of loss wrt alpha
            gradients = tape.gradient(total_loss, trainable_variables)

            # Update the alpha variables
            optimizer.apply_gradients(zip(gradients, trainable_variables))

        log = f"Epoch {e + 1}"
        log += f" - loss (quantization|rounding) {q_loss:.8f} | {r_loss:.8f}"
        for adr in adarounds:
            roamers, floor_r, ceil_r = adr.status()
            log += f" - roamers: {roamers} - ↓↑ [{floor_r:.2f}, {ceil_r:.2f}]"
        print_console(log, new_line=False)
        if beta > 0 and r_loss == 0:
            # All weights have been rounded
            break
    # Go to the next line in console
    print_console("")
    # Set Adaround quantizer to evaluation mode
    for adr in adarounds:
        adr.training = False
    # Evaluate final quantization error
    err_after = sampler.quantization_error(m)
    print_console(f"quantization error: {err_before:.4f} -> {err_after:.4f}")
    # Freeze optimized weights
    finalize_training(layer)


def adaround(model,
             samples,
             optimizer,
             epochs,
             loss=keras.losses.MeanSquaredError(),
             batch_size=None,
             include_activation=False):
    """Optimize the rounding of quantized weights.

    This implements the Adaround algorithm described in:
    Up or Down? Adaptive Rounding for Post-Training Quantization
    Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen
    Blankevoort
    https://arxiv.org/abs/2004.10568

    Instead of rounding weights to the nearest, Adaround introduces a tensor of
    continuous variables representing the decimals of the float weights, and
    thus formulates the minimization of the quantization error as a  Quadratic
    Unconstrained Binary Optimization problem, iteratively pushing the decimal
    variables to a distribution of 0 and 1 minimizing the error.

    After the optimization, the quantization scales are preserved, but each
    weight is closer or equal to a quantized value.

    When optimizing a model, the following must be provided:

    - a set of samples (typically from the training dataset),
    - an optimizer,
    - the maximum number of epochs (the optimization of a layer stops when all
      weights have been rounded).

    Args:
        model (:obj:`keras.Model`): a quantized Keras Model
        samples (:obj:`np.ndarray`): a set of samples used for calibration
        optimizer (:obj:`tensorflow.keras.optimizers.Optimizer`): an optimizer
        epochs (int): the maximum number of epochs
        loss (:obj:`tensorflow.keras.losses.Loss`): the error loss function
        batch_size (int): the batch size used when evaluating samples
        include_activation (bool): quantization error is evaluated after
          activation.

    Returns:
        :obj:`tf.Model`: a quantized Keras model whose weights have been
        optimized
    """
    adaround_model = clone_model_with_weights(model)
    # Initialize all trainable variables
    adarounds, trainable_variables = initialize_training(optimizer, adaround_model)

    for layer in adaround_model.layers:
        if is_quantized_neural(layer):
            # Optimize rounding
            optimize_rounding(
                adaround_model,
                samples,
                layer,
                optimizer,
                epochs,
                loss,
                batch_size,
                include_activation,
                adarounds[layer],
                trainable_variables[layer])
    return adaround_model
