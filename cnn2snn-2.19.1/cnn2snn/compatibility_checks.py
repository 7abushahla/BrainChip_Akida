# ******************************************************************************
# Copyright 2021 Brainchip Holdings Ltd.
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
"""Functions to check model compatibility for CNN2SNN conversion.
"""

from tf_keras import Sequential, layers

from . import quantization_layers as qlayers
from .transforms.sequential import prepare_to_convert

conv_neural_layers = (layers.Conv2D, layers.SeparableConv2D, qlayers.QuantizedConv2D,
                      qlayers.QuantizedSeparableConv2D)

dense_neural_layers = (layers.Dense, qlayers.QuantizedDense)

neural_layers = conv_neural_layers + dense_neural_layers


def check_model_compatibility(model):
    r"""Checks if a Keras model is compatible for cnn2snn conversion.

    This function doesn't convert the Keras model to an Akida model
    but only checks if the model design is compatible.

    Note that this function doesn't check if the model is compatible with
    Akida hardware.
    To check compatibility with a specific hardware device, convert the model
    and call `model.map` with this device as argument.

    **1. How to build a compatible Keras quantized model?**

    The following lines give details and constraints on how to build a Keras
    model compatible for the conversion to an Akida model.


    **2. General information about layers**

    An Akida layer must be seen as a block of Keras layers starting with a
    processing layer (Conv2D, SeparableConv2D,
    Dense). All blocks of Keras layers except the last block must have
    exactly one activation layer (ReLU or ActivationDiscreteRelu). Other
    optional layers can be present in a block such as a pooling layer or a
    batch normalization layer.
    Here are all the supported Keras layers for an Akida-compatible model:

    - Processing layers:

      - keras Conv2D/SeparableConv2D/Dense
      - cnn2snn QuantizedConv2D/QuantizedSeparableConv2D/QuantizedDense

    - Activation layers:

      - keras ReLU
      - cnn2snn ActivationDiscreteRelu
      - any increasing activation function (only for the last block of layers)
        such as softmax, sigmoid set as last layer. This layer must derive from
        keras.layers.Activation, and it will be removed during conversion to
        an Akida model.

    - Pooling layers:

      - MaxPool2D
      - GlobalAvgPool2D

    - BatchNormalization
    - Dropout
    - Flatten
    - Input
    - Reshape

    Example of a block of Keras layers::

              ----------
              | Conv2D |
              ----------
                  ||
                  \/
        ----------------------
        | BatchNormalization |
        ----------------------
                  ||
                  \/
             -------------
             | MaxPool2D |
             -------------
                  ||
                  \/
       --------------------------
       | ActivationDiscreteRelu |
       --------------------------


    **3. Constraints about inputs**

    An Akida model can accept two types of inputs: sparse events or 8-bit
    images. Whatever the input type, the Keras inputs must respect the
    following relation:

        input_akida = scale * input_keras + shift

    where the Akida inputs must be positive integers, the input scale must be
    a float value and the input shift must be an integer. In other words,
    scale * input_keras must be integers.

    Depending on the input type:

    - if the inputs are events (sparse), the first layer of the Keras model can
      be any processing layer. The input shift must be zero.
    - if the inputs are images, the first layer must be a Conv2D
      layer.


    **4. Constraints about layers' parameters**

    To be Akida-compatible, the Keras layers must observe the following rules:

    - all layers with the 'data_format' parameter must be 'channels_last'
    - all processing quantized layers and ActivationDiscreteRelu must have a
      valid quantization bitwidth
    - a Dense layer must have an input shape of (N,) or (1, 1, N)
    - a BatchNormalization layer must have 'axis' set to -1 (default)
    - a BatchNormalization layer cannot have negative gammas
    - Reshape layers can only be used to transform a tensor of shape (N,) to a
      tensor of shape (1, 1, N), and vice-versa
    - only one pooling layer can be used in each block
    - a MaxPool2D layer must have the same 'padding' as the corresponding
      processing quantized layer

    **5. Constraints about the order of layers**

    To be Akida-compatible, the order of Keras layers must observe the following
    rules:

    - a block of Keras layers must start with a processing quantized layer
    - where present, a BatchNormalization/GlobalAvgPool2D layer must be placed
      before the activation
    - a Flatten layer can only be used before a Dense layer
    - an Activation layer other than ReLU can only be used in the last layer


    Args:
        model (:obj:`keras.Model`): the model to check.
    """
    try:
        # Prepare the model with appropriate transformations
        sync_model = prepare_to_convert(model)

        # Check model compatibility
        check_sequential_compatibility(sync_model)
        return True
    except RuntimeError as e:
        print(
            "The Keras quantized model is not compatible for a conversion "
            "to an Akida model:\n", str(e))
        return False


def check_sequential_compatibility(model):
    """Checks compatibility of a Sequential Keras model.

    If an incompatibility is detected, an error is raised. This function can be
    applied on native Keras models and does not check quantization parameters.

    Args:
        model (:obj:`keras.Model`): a Sequential Keras model.
    """

    assert isinstance(model, Sequential)

    if len(model.layers) == 0:
        return

    # The two following variables are used to know if there is an alternation
    # between a neural layer and an activation layer. For example, when we go
    # through a neural layer in the model, 'last_visited_neural_layer' must be
    # None. If not, it means that there is no activation layer between this
    # neural layer and the previous one. 'last_visited_neural_layer' is then set
    # to the current neural layer and 'last_visited_activation' is set to None.
    # A (sub)model must start with a neural layer.
    last_visited_neural_layer = None
    last_visited_activation = model.layers[0]

    for i, layer in enumerate(model.layers):
        if type(layer) in neural_layers:
            _check_dense_shape(layer)
            _check_conv_params(layer)
            _check_two_neural_layers(layer, last_visited_neural_layer)
            # Update last visited neural and activation layer
            last_visited_neural_layer = layer
            last_visited_activation = None
        elif isinstance(layer, (layers.ReLU, qlayers.QuantizedActivation)):
            _check_two_successive_activations(layer, last_visited_activation)
            # Update last visited neural and activation layer
            last_visited_neural_layer = None
            last_visited_activation = layer
        elif isinstance(layer, (layers.MaxPool2D, layers.GlobalAvgPool2D)):
            _check_pooling_compatibility(model, i)
        elif isinstance(layer, layers.Flatten):
            _check_flatten_layer(model, i)
        elif isinstance(layer, layers.Reshape):
            _check_reshape_layer(layer)
        # Rescaling layer only supported as first layer of the first submodel
        elif isinstance(layer, layers.Rescaling):
            _check_rescaling_compatibility(layer, i)
        # Activation/Softmax layer only supported after the last neural layer of
        # the last submodel
        elif isinstance(layer, (layers.Activation, layers.Softmax)):
            _check_unsupported_activation(model, i)
        else:
            raise RuntimeError(
                f"Layer {layer.name} of type {layer.__class__.__name__} is not "
                f"supported for Akida conversion")


def _check_dense_shape(layer):
    """Asserts Dense layer is compatible for conversion.
    One check is performed here:
    - input shape must be (bs, N) or (bs, 1, 1, N) (bs is the batch size).

    Args:
        layer(:obj:`keras.Layer`): the Dense or QuantizedDense layer to
            check.
    """

    if type(layer) not in dense_neural_layers:
        return

    # Raises error if Dense input shape is incorrect: supported
    # shapes are (N,) and (1, 1, N). Remember input_shape has the batch
    # size as first element of tuple.
    # Input shape is (1, 1, N) and (N,)
    valid = (len(layer.input_shape) == 2 or (len(layer.input_shape) ==
             4 and layer.input_shape[1] == 1 and layer.input_shape[2] == 1))
    if not valid:
        raise RuntimeError(
            f"The Dense layer {layer.name} must have an input shape of (N,) or (1,1,N). Receives "
            f"{layer.input_shape[1:]}.")


def _check_conv_params(layer):
    """Asserts convolution layer is compatible for conversion.

    One check is performed here:
    - dilation_rate must have a default value.

    Args:
        layer(:obj:`keras.Layer`): the convolution layer to check.
    """

    if type(layer) not in conv_neural_layers:
        return

    # Raises error if the layer has dilation rate.
    if layer.dilation_rate != (1, 1):
        raise ValueError(f"The layer {layer.name} must have a dilation rate of 1. Receives "
                         f"{layer.dilation_rate}.")


def _check_two_neural_layers(layer, last_visited_neural_layer):
    """Checks that there is an activation between two neural layers.

    Args:
        layer (:obj:`keras.Layer`): the current neural layer.
        last_visited_neural_layer (:obj:`keras.Layer`): the last visited
            neural layer if there is no activation between this last visited and
            the current neural layer. If there is an activation, this argument
            must be None.
    """

    if last_visited_neural_layer is not None:
        raise RuntimeError(
            f"An activation layer is required between the two neural layers "
            f"'{last_visited_neural_layer.name}' and '{layer.name}'.")


def _check_two_successive_activations(layer, last_visited_activation):
    """Checks that there is a neural layer between two activation layers.

    Args:
        layer (:obj:`keras.Layer`): the current activation layer.
        last_visited_activation (:obj:`keras.Layer`): the last visited
            activation layer if there is no neural layer between this last
            visited and the current activation. If there is a neural layer, this
            argument must be None.
    """
    if last_visited_activation is not None:
        raise RuntimeError(
            f"A neural layer is required between the two activation layers "
            f"'{last_visited_activation.name}' and '{layer.name}'.")


def _check_pooling_compatibility(model, pool_index):
    """Asserts pooling layer is compatible for conversion. Transformations must
    have been applied before these checks.
    Two checks are performed here:
    - a pooling layer must be placed directly after a neural layer
    - the padding of MaxPool2D must be the same as the padding of neural layer

    Args:
        model (:obj:`keras.Model`): the Sequential model to check.
        pool_index (int): the index of the pooling layer to check.
    """

    layer_pool = model.layers[pool_index]
    prev_layer = model.layers[pool_index - 1]

    # Raise error if pooling layer is the first layer of the sequence/branch
    if pool_index == 0:
        raise RuntimeError(f"Pooling layer '{layer_pool.name}' cannot be the "
                           "first layer of a model or sequence. It must be "
                           "placed after a convolutional layer.")

    # Raise error if GlobalAvgPool2D is placed after the activation
    if type(prev_layer) not in conv_neural_layers:
        raise RuntimeError(f"Pooling layer {layer_pool.name} must be placed "
                           "after a convolutional neural layer. Currently after"
                           f" {prev_layer.name}.")

    # Raises error if the padding of MaxPool2D is different from the padding
    # of the neural processing layer.
    if (isinstance(layer_pool, layers.MaxPool2D) and prev_layer.padding != layer_pool.padding):
        raise RuntimeError(f"Pooling layer {layer_pool.name} (padding: "
                           f"{layer_pool.padding}) must have the same "
                           f"padding as {prev_layer.name} (padding: "
                           f"{prev_layer.padding}).")


def _check_flatten_layer(model, flatten_index):
    """Checks that the Flatten layer is supported for conversion.
    A Flatten layer is supported if it is followed by a Dense layer or if it
    is the last layer of the model.

    Args:
        model (:obj:`keras.Model`): a Sequential Keras (sub)model.
        flatten_index (int): the index of the Flatten layer in the model's
            layers.
    """

    try:
        if type(model.layers[flatten_index + 1]) not in (layers.Dense, qlayers.QuantizedDense):
            raise RuntimeError(
                "A Flatten layer is only supported before a Dense one. Receives"
                f" unsupported Flatten layer '{model.layers[flatten_index]}'.")
    except IndexError:
        pass


def _check_reshape_layer(layer):
    """This function checks if the Reshape layer is supported.
    In the cnn2snn conversion, a Reshape layer can only be used to transform
    a tensor of shape (N,) to a tensor of shape (1, 1, N), and vice-versa.
    Note that the 'input_shape' and 'output_shape' parameters of a layer has
    the batch size as first element:
        input_shape = (batch_size,) + input_tensor_shape
    The batch size is ignored in the following function.

    Args:
        layer(:obj:`keras.Layer`): the Reshape layer to check.
    """
    in_shape = layer.input_shape
    out_shape = layer.output_shape

    valid = ((  # Reshape from (1,1,N) to (N,)
        # Reshape from (N,) to (1,1,N)
        # Useless Reshape, from X to X
        len(in_shape) == 4 and in_shape[1] == 1 and in_shape[2] == 1
        and len(out_shape) == 2 and out_shape[1] == in_shape[3])
        or (len(in_shape) == 2 and len(out_shape) == 4 and out_shape[1] == 1 and out_shape[2] == 1
            and out_shape[3] == in_shape[1]) or (in_shape == out_shape))

    if not valid:
        raise RuntimeError(
            f"The Reshape layer {layer.name} can only be used to transform a tensor of shape (N,) "
            "to a tensor of shape (1, 1, N), and vice-versa. Receives "
            f"input_shape {in_shape[1:]} and output_shape "
            f"{out_shape[1:]}.")


def _check_rescaling_compatibility(layer, index):
    """Checks that Rescaling layer is only accepted as the first layer.

    Args:
        layer(:obj:`keras.Layer`): the Rescaling layer to check.
        index (int): the index of the Rescaling layer in the model.
    """

    if index != 0:
        raise RuntimeError("A Rescaling layer must be the first layer of the "
                           f"model. Receives unexpected layer {layer.name}.")


def _check_unsupported_activation(model, index):
    """Checks that an unsupported activation layer (keras.Activation or
    keras.Softmax) is only present after the last neural layer of the last
    submodel.

    Args:
        model(:obj:`keras.Model`): the model to check.
        index (int): the index of the unsupported activation layer in the model.
    """

    last_layers_type = {type(layer) for layer in model.layers[index + 1:]}
    if last_layers_type.intersection(neural_layers):
        raise RuntimeError(
            "Activation layers other than ReLU and quantized activations are "
            "not supported before a neural layer. Receives activation "
            f"layer '{model.layers[index].name}' before a neural layer.")
