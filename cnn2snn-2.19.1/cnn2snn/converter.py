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
"""Conversion of a Keras/CNN2SNN model into an Akida model"""

import os
import warnings
import tensorflow as tf
import tf_keras as keras
from onnx import ModelProto
from .model_generator import generate_model as cnn2snn_generate_model
from .quantizeml import generate_model as qml_generate_model

import akida
from .akida_versions import get_akida_version, AkidaVersion
from .transforms import fix_v1_activation_variables, prepare_to_convert
from .compatibility_checks import check_sequential_compatibility, _collect_sequential_issues
from quantizeml.models import QuantizationParams, quantize as quantize_qml


def check_model_compatibility(model, device=None, input_dtype="uint8"):
    """Checks that a float Keras or ONNX model is Akida compatible.

    The process stops on the first incompatibility encountered with an exception. The
    problematic step (quantization or conversion or mapping) is indicated in the exception message.
    Then if errors occurs, issues must be fixed iteratively in order to obtain an Akida
    compatible model.
    Note that the version context is used to determine compatibility.

    Args:
        model (:obj:`keras.Model` or :obj:`onnx.ModelProto`): the model to check.
        device (:obj:`akida.HwDevice`, optional): the device to map on. If a device is provided,
            there will be a check that the model can fully run on such device. Defaults to None.
        input_dtype (np.dtype or str, optional): expected model input format.
            If given as a string, should follow numpy string type requirements. Defaults to 'uint8'.

    Raises:
        ValueError: if model type is incompatible with Akida version context.
        ValueError: if device is incompatible with Akida version context.
        Exception: if an incompatibility is encountered on quantization/conversion/mapping steps.
    """
    # Preliminary checks:
    # Onnx models are not supported on Akida v1
    if (get_akida_version() == AkidaVersion.v1) and (isinstance(model, ModelProto)):
        raise ValueError("Akida v1 does not support ONNX models. Use Akida v2 instead.")

    # if a device is provided, it should be consistent with the Akida version used.
    if device is not None and device.ip_version == akida.IpVersion.v1 \
            and (get_akida_version() != AkidaVersion.v1):
        raise ValueError(f"The device {device} is inconsistent with the Akida version used.")

    # Quantization step
    if get_akida_version() == AkidaVersion.v1:
        # 4 bits weights and activation per tensor
        q_param = QuantizationParams(activation_bits=4, input_weight_bits=4, weight_bits=4,
                                     per_tensor_activations=True, input_dtype=input_dtype)
    else:
        q_param = QuantizationParams(input_dtype=input_dtype)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_q = quantize_qml(model, qparams=q_param, num_samples=1)
    except Exception as e:
        raise type(e)('Incompatibility found during quantization: ' + str(e))

    # Conversion step
    try:
        model_ak = convert(model_q)
    except Exception as e:
        raise type(e)('Incompatibility found during conversion: ' + str(e))

    if device is not None:
        # Mapping step
        try:
            model_ak.map(device, hw_only=True)
        except Exception as e:
            raise type(e)('Incompatibility found during mapping: ' + str(e))


def _print_issues_report(issues):
    """Pretty-print the list of issues collected by check_model_compatibility_all."""
    if not issues:
        print("✅  No compatibility issues found. Model looks good!")
        return
    print(f"❌  Found {len(issues)} compatibility issue(s):\n")
    for idx, issue in enumerate(issues, 1):
        print(f"  [{idx}] {issue}")
    print()


def _check_static_v1_constraints(model):
    """Check static Akida v1 constraints that aren't fully enforced elsewhere.

    This covers things like input dimensions and Dense fan-in size so that
    a model passing this check has a much higher chance to fully run on v1
    hardware, not just convert.
    """
    issues = []

    # Only apply to Keras models in Akida v1 context
    if get_akida_version() != AkidaVersion.v1 or isinstance(model, ModelProto):
        return issues

    # --- Input dimensions: width ≤ 256, height ≥ 5, channels in {1, 3} ---
    try:
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
            # Functional models sometimes expose a list of input shapes
            input_shape = input_shape[0]

        if input_shape is not None:
            # Accept (batch, H, W, C) or (H, W, C)
            if len(input_shape) == 4:
                _, h, w, c = input_shape
            elif len(input_shape) == 3:
                h, w, c = input_shape
            else:
                h = w = c = None

            if w is not None and w > 256:
                issues.append(
                    f"[INPUT] Input width {w} exceeds Akida v1 limit of 256 pixels.")
            if h is not None and h < 5:
                issues.append(
                    f"[INPUT] Input height {h} is below Akida v1 minimum of 5 pixels.")
            if c is not None and c not in (1, 3):
                issues.append(
                    f"[INPUT] Input channels must be 1 or 3 for Akida v1. Receives {c}.")
        else:
            issues.append("[INPUT] Could not determine input_shape for model.")
    except Exception as e:  # pragma: no cover - defensive
        issues.append(f"[INPUT] Failed to inspect input_shape: {e}")

    # --- Dense fan-in: total input features ≤ 57,334 ---
    try:
        for layer in getattr(model, "layers", []):
            if isinstance(layer, keras.layers.Dense):
                in_shape = getattr(layer, "input_shape", None)
                if not in_shape:
                    continue
                # (batch, N) or (batch, 1, 1, N)
                if len(in_shape) == 2:
                    features = in_shape[-1]
                elif len(in_shape) == 4 and in_shape[1] == 1 and in_shape[2] == 1:
                    features = in_shape[-1]
                else:
                    # Non-standard Dense input shapes will already be caught by
                    # structural checks; no need to duplicate here.
                    continue

                if features is not None and features > 57334:
                    issues.append(
                        f"[DENSE] Dense layer '{layer.name}' has {features} input features, "
                        "exceeding the Akida v1 guideline of 57,334.")
    except Exception as e:  # pragma: no cover - defensive
        issues.append(f"[DENSE] Failed to inspect Dense fan-in sizes: {e}")

    return issues


def check_model_compatibility_all(model, device=None, input_dtype="uint8"):
    """Checks Akida compatibility and reports ALL issues at once.

    Unlike :func:`check_model_compatibility`, which stops at the first error,
    this function runs every check it can independently and accumulates all
    problems into a single report so you can fix them all in one go.

    Checks performed (in order):
      1. Preliminary version/type checks (ONNX on v1, device/version mismatch).
      2. Per-layer structural checks on every layer (exhaustive walk).
      3. Quantization with QuantizeML (one attempt, error captured if any).
      4. CNN2SNN conversion (only attempted if quantization succeeded).
      5. Hardware mapping (only if device provided and conversion succeeded).

    Args:
        model (:obj:`keras.Model` or :obj:`onnx.ModelProto`): the model to check.
        device (:obj:`akida.HwDevice`, optional): hardware device to test mapping
            against. Defaults to None.
        input_dtype (np.dtype or str, optional): expected input dtype.
            Defaults to 'uint8'.

    Returns:
        list[str]: all issues found (empty list means fully compatible).
    """
    issues = []

    # ------------------------------------------------------------------ #
    # 1. Preliminary checks
    # ------------------------------------------------------------------ #
    if (get_akida_version() == AkidaVersion.v1) and isinstance(model, ModelProto):
        issues.append(
            "[PRELIMINARY] Akida v1 does not support ONNX models. Use Akida v2 instead.")

    if device is not None and device.ip_version == akida.IpVersion.v1 \
            and (get_akida_version() != AkidaVersion.v1):
        issues.append(
            f"[PRELIMINARY] Device '{device}' is an Akida v1 device but the active "
            f"version context is not AkidaVersion.v1.")

    # Bail out early – subsequent checks would be meaningless or crash
    if issues:
        _print_issues_report(issues)
        return issues

    # ------------------------------------------------------------------ #
    # 2. Per-layer structural checks (exhaustive – never stops early)
    # ------------------------------------------------------------------ #
    try:
        sync_model = prepare_to_convert(model)
        layer_issues = _collect_sequential_issues(sync_model)
        for msg in layer_issues:
            issues.append(f"[STRUCTURE] {msg}")
    except Exception as e:
        issues.append(f"[STRUCTURE] prepare_to_convert failed: {e}")

    # ------------------------------------------------------------------ #
    # 2b. Static Akida v1-specific constraints (input dims, Dense fan-in)
    # ------------------------------------------------------------------ #
    static_issues = _check_static_v1_constraints(model)
    issues.extend(static_issues)

    # ------------------------------------------------------------------ #
    # 3. Quantization check
    # ------------------------------------------------------------------ #
    if get_akida_version() == AkidaVersion.v1:
        q_param = QuantizationParams(
            activation_bits=4, input_weight_bits=4, weight_bits=4,
            per_tensor_activations=True, input_dtype=input_dtype)
    else:
        q_param = QuantizationParams(input_dtype=input_dtype)

    model_q = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_q = quantize_qml(model, qparams=q_param, num_samples=1)
    except Exception as e:
        issues.append(f"[QUANTIZATION] {e}")

    # ------------------------------------------------------------------ #
    # 4. Conversion check
    # ------------------------------------------------------------------ #
    model_ak = None
    if model_q is not None:
        try:
            model_ak = convert(model_q)
        except Exception as e:
            issues.append(f"[CONVERSION] {e}")
    else:
        issues.append("[CONVERSION] Skipped — quantization failed.")

    # ------------------------------------------------------------------ #
    # 5. Hardware mapping check
    # ------------------------------------------------------------------ #
    if device is not None:
        if model_ak is not None:
            try:
                model_ak.map(device, hw_only=True)
            except Exception as e:
                issues.append(f"[MAPPING] {e}")
        else:
            issues.append("[MAPPING] Skipped — conversion failed.")

    _print_issues_report(issues)
    return issues


def convert(model, file_path=None, input_scaling=None):
    """Converts a Keras or ONNX quantized model to an Akida one.

    This method is compatible with model quantized with :func:`cnn2snn.quantize`
    and :func:`quantizeml.quantize`. To check the difference between the two
    conversion processes check the methods _convert_cnn2snn and _convert_quantizeml
    below.

    Args:
        model (:obj:`keras.Model` or :obj:`onnx.ModelProto`): a model to convert.
        file_path (str, optional): destination for the akida model.
            (Default value = None)
        input_scaling (2 elements tuple, optional): value of the input scaling.
            (Default value = None)

    Returns:
        :obj:`akida.Model`: an Akida model.
    """
    if not tf.executing_eagerly():
        raise SystemError("Tensorflow eager execution is disabled. "
                          "It is required to convert Keras weights to Akida.")

    # Check if the model has been quantized with quantizeml by checking quantized layers type
    # or with ONNX tools
    cnn2snn_model = not (isinstance(model, ModelProto) or
                         any("quantizeml" in str(type(layer)) for layer in model.layers))

    # Convert the model
    if cnn2snn_model:
        ak_model = _convert_cnn2snn(model, input_scaling)
    else:
        if input_scaling:
            warnings.warn("Cannot use input_scaling parameter when converting QuantizeML models "
                          "to Akida. Continuing execution without this parameter.")
        ak_model = _convert_quantizeml(model)

    # Perform post conversion checks on the model
    ak_model = _post_conversion_checks(ak_model)

    # Save model if file_path is given
    if file_path:
        # Create directories
        dir_name, base_name = os.path.split(file_path)
        if base_name:
            file_root, file_ext = os.path.splitext(base_name)
            if not file_ext:
                file_ext = '.fbz'
        else:
            file_root = model.name
            file_ext = '.fbz'

        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        save_path = os.path.join(dir_name, file_root + file_ext)
        ak_model.save(save_path)

    return ak_model


def _convert_quantizeml(model):
    """Converts a Keras or ONNX quantized model with quantizeml to an Akida one.

    After quantizing a Keras model with :func:`quantizeml.quantize`, it can be
    converted to an Akida model.

    Args:
        model (:obj:`keras.Model` or :obj:`onnx.ModelProto`): the model to convert.

    Returns:
        :obj:`akida.Model`: an Akida model.

    """

    # Generate Akida model with empty weights/thresholds for now
    ak_model = qml_generate_model(model)

    return ak_model


def _convert_cnn2snn(model, input_scaling=None):
    """Converts a Keras quantized model to an Akida one.

    After quantizing a Keras model with :func:`cnn2snn.quantize`, it can be
    converted to an Akida model. By default, the conversion expects that the
    Akida model takes 8-bit images as inputs. ``input_scaling`` defines how the
    images have been rescaled to be fed into the Keras model (see note below).

    If inputs are spikes, Akida inputs are then expected to be integers between
    0 and 15.

    Note:
        The relationship between Keras and Akida inputs is defined as::

            input_akida = input_scaling[0] * input_keras + input_scaling[1].

        If a :class:`keras.layers.Rescaling`
        layer is present as first layer of the model, ``input_scaling`` must
        be None: the :class:`Rescaling` parameters will be used to compute the
        input scaling.

    Examples:

        >>> # Convert a quantized Keras model with Keras inputs as images
        >>> # rescaled between -1 and 1
        >>> inputs_akida = images.astype('uint8')
        >>> inputs_keras = (images.astype('float32') - 128) / 128
        >>> model_akida = cnn2snn.convert(model_keras, input_scaling=(128, 128))
        >>> model_akida.predict(inputs_akida)

        >>> # Convert a quantized Keras model with Keras inputs as spikes and
        >>> # input scaling of (2.5, 0). Akida spikes must be integers between
        >>> # 0 and 15
        >>> inputs_akida = spikes.astype('uint8')
        >>> inputs_keras = spikes.astype('float32') / 2.5
        >>> model_akida = cnn2snn.convert(model_keras, input_scaling=(2.5, 0))
        >>> model_akida.predict(inputs_akida)

        >>> # Convert and directly save the Akida model to fbz file.
        >>> cnn2snn.convert(model_keras, 'model_akida.fbz')

    Args:
        model (:obj:`keras.Model`): a keras model
        input_scaling (2 elements tuple, optional): value of the input scaling.
            (Default value = None)

    Returns:
        :obj:`akida.Model`: an Akida model.

    Raises:
        ValueError: If ``input_scaling[0]`` is null or negative.
        ValueError: If a :class:`Rescaling` layer is present and
            ``input_scaling`` is not None.
        SystemError: If Tensorflow is not run in eager mode.
    """

    # Check Keras Rescaling layer to replace the input_scaling
    rescaling_input_scaling = _get_rescaling_layer_params(model)
    if rescaling_input_scaling is not None and input_scaling is not None:
        raise ValueError("If a Rescaling layer is present in the model, "
                         "'input_scaling' argument must be None. Receives "
                         f"{input_scaling}.")

    input_scaling = rescaling_input_scaling or input_scaling or (1, 0)

    if input_scaling[0] <= 0:
        raise ValueError("The scale factor 'input_scaling[0]' must be strictly"
                         f" positive. Receives: input_scaling={input_scaling}")

    # Prepare model for conversion
    sync_model = prepare_to_convert(model)

    # Check model compatibility
    check_sequential_compatibility(sync_model)

    # Generate Akida model with converted weights/thresholds
    ak_model = cnn2snn_generate_model(sync_model, input_scaling)

    return ak_model


def _get_rescaling_layer_params(model):
    """Computes the new input scaling retrieved from the Keras
    `Rescaling` layer.

    Keras Rescaling layer works as:

     input_k = scale * input_ak + offset

    CNN2SNN input scaling works as:

     input_ak = input_scaling[0] * input_k + input_scaling[1]

    Equivalence leads to:

     input_scaling[0] = 1 / scale
     input_scaling[1] = -offset / scale

    Args:
        model (:obj:`keras.Model`): a keras model.

    Returns:
        tuple: the new input scaling from the Rescaling layer or None if
            no Rescaling layer is at the beginning of the model.
    """

    Rescaling = keras.layers.Rescaling
    for layer in model.layers[:2]:
        if isinstance(layer, Rescaling):
            return (1 / layer.scale, -layer.offset / layer.scale)
    return None


def _post_conversion_checks(model):
    """ Perform post-conversion checks on an akida model.

    The model is not necessarily changed and can be the original one.

    Args:
        model (Model): the model to check
    """
    if model.ip_version == akida.core.IpVersion.v1:
        # check if the akida v1 model has an invalid act_step, that prevents the model
        # to map on HW. If so equalize the Akida model activation variables.
        fix_v1_activation_variables(model)
    return model
