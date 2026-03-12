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
"""Command-line interface"""

import argparse
import re
import os

import tf_keras as keras
import numpy as np
import onnx

from .converter import convert
from .quantization import quantize
from .utils import load_quantized_model
from .transforms import reshape, normalize_separable_model
from .calibration import adaround, bias_correction, activations_rescaling


def quantize_model(model_path, wq, aq, iq, fold_BN):
    """Wrapper to quantize model"""
    model = load_quantized_model(model_path)
    if iq == -1:
        iq = wq
    model_q = quantize(model, wq, aq, iq, fold_BN)
    # Extract base name
    base_name = os.path.splitext(model_path)[0]
    # Cross-platform path may contain alpha characters, /, \ and :
    path_re = r"([\w/\\:~]+)"
    # Quantization suffix has a well-identified structure
    suffix_re = r"(_iq\d_wq\d_aq\d)"
    p = re.compile(path_re + suffix_re)
    # Look for an existing quantization suffix in the base name
    m = p.match(base_name)
    if m:
        # Only keep the actual base name (group(2) contains the suffix)
        base_name = m.group(1)
    out_path = f"{base_name}_iq{iq}_wq{wq}_aq{aq}.h5"
    model_q.save(out_path, include_optimizer=False)
    print(f"Model successfully quantized and saved as {out_path}.")


def convert_model(model_path, input_scaling):
    """Wrapper to convert model"""
    base_name, ext = os.path.splitext(model_path)
    if ext.lower() == ".onnx":
        q_model = onnx.load_model(model_path)
    else:
        q_model = load_quantized_model(model_path)
    out_path = f"{base_name}.fbz"
    convert(q_model, input_scaling=input_scaling, file_path=out_path)
    print(f"Model successfully converted and saved as {out_path}.")


def reshape_model(model_path, input_height, input_width):
    """Wrapper to reshape model"""
    # Load the original model file
    base_model = load_quantized_model(model_path)

    # Extract base name
    base_name = os.path.splitext(model_path)[0]

    model = reshape(base_model, input_height, input_width)

    out_path = f"{base_name}_{input_height}_{input_width}.h5"

    model.save(out_path)
    print(f"Model input successfully reshaped and saved as {out_path}.")


def calibrate_adaround_model(base_model, base_name, samples, batch_size, epochs,
                             lr, include_act):
    """Wrapper to Adaround calibration model method"""

    out_path = f"{base_name}_adaround_calibrated.h5"

    # Apply Adaround
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    adaround_model = adaround(base_model,
                              samples,
                              optimizer,
                              epochs=epochs,
                              loss=keras.losses.MeanSquaredError(),
                              batch_size=batch_size,
                              include_activation=include_act)

    adaround_model.save(out_path)
    print(f"Model successfully calibrated with Adaround method and saved \
            as {out_path}.")


def calibrate_bias_correction_model(base_model, base_name, samples, batch_size):
    """Wrapper to Bias Correction calibration model method"""

    out_path = f"{base_name}_bc_calibrated.h5"

    # Apply bias correction
    bc_model = bias_correction(base_model, samples, batch_size)

    bc_model.save(out_path)
    print(f"Model successfully calibrated with Bias Correction method and \
            saved as {out_path}.")


def main():
    """CNN2SNN command-line interface to quantize/convert/upgrade a model"""
    parser = argparse.ArgumentParser()
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-m",
                               "--model",
                               type=str,
                               required=True,
                               help="The source model")
    sp = parser.add_subparsers(dest="action")
    q_parser = sp.add_parser("quantize",
                             parents=[parent_parser],
                             help="Quantize a Keras model")
    q_parser.add_argument("-wq",
                          "--weight_quantization",
                          type=int,
                          default=4,
                          help="The weight quantization")
    q_parser.add_argument("-aq",
                          "--activ_quantization",
                          type=int,
                          default=4,
                          help="The activations quantization")
    q_parser.add_argument("-iq",
                          "--input_weight_quantization",
                          type=int,
                          default=-1,
                          help="The first layer weight quantization (same as"
                          " weight_quantization if omitted)")
    q_parser.add_argument("--no-fold-BN",
                          dest="fold_BN",
                          action='store_false',
                          help="If specified, do not fold BatchNormalization "
                          "layers before quantization")
    c_parser = sp.add_parser(
        "convert",
        parents=[parent_parser],
        help="Convert a quantized Keras/ONNX model to an Akida model")
    c_parser.add_argument("-sc",
                          "--scale",
                          type=int,
                          default=None,
                          help="The scale factor applied on uint8 inputs.")
    c_parser.add_argument("-sh",
                          "--shift",
                          type=int,
                          default=None,
                          help="The shift applied on uint8 inputs.")
    r_parser = sp.add_parser(
        "reshape",
        parents=[parent_parser],
        help="Reshape a (quantized) Keras model Input layer to a given size.")
    r_parser.add_argument("-ih",
                          "--input_height",
                          type=int,
                          default=None,
                          required=True,
                          help="The new input height.")
    r_parser.add_argument("-iw",
                          "--input_width",
                          type=int,
                          default=None,
                          required=True,
                          help="The new input width.")
    calib_parser = sp.add_parser("calibrate",
                                 help="Calibrate a quantized Keras model")
    # Create parent subparser for arguments shared between calibrate methods
    calib_parent = argparse.ArgumentParser(add_help=False,
                                           parents=[parent_parser])
    calib_parent.add_argument(
        "-sa",
        "--samples",
        type=str,
        default=None,
        required=True,
        help="Set of samples to calibrate the model parameters (.npz file).")
    calib_parent.add_argument("-bs",
                              "--batch_size",
                              type=int,
                              default=None,
                              help="The batch size.")
    calib_method_parser = calib_parser.add_subparsers(
        dest="method",
        help="Weight calibration method. Bias Correction or AdaRound.")
    calib_method_parser.add_parser(
        "bc",
        help="Bias Correction weight calibration method.",
        parents=[calib_parent])
    adaround_parser = calib_method_parser.add_parser(
        "adaround",
        help="Adaround weight calibration method.",
        parents=[calib_parent])
    adaround_parser.add_argument("-e",
                                 "--epochs",
                                 type=int,
                                 default=10,
                                 help="The number of epochs.")
    adaround_parser.add_argument("-lr",
                                 "--learning_rate",
                                 type=float,
                                 default=1e-2,
                                 help="The learning rate.")
    adaround_parser.add_argument("-act",
                                 "--include_activation",
                                 action='store_true',
                                 help="Include activation or not.")

    args = parser.parse_args()
    if args.action == "quantize":
        quantize_model(args.model, args.weight_quantization,
                       args.activ_quantization, args.input_weight_quantization,
                       args.fold_BN)
    if args.action == "convert":
        if args.scale is None or args.shift is None:
            input_scaling = None
        else:
            input_scaling = (args.scale, args.shift)
        convert_model(args.model, input_scaling)
    if args.action == "reshape":
        reshape_model(args.model, args.input_height, args.input_width)
    if args.action == "calibrate":
        data = np.load(args.samples)
        lst = data.files
        samples_arr = data[lst[0]]
        for item in lst[1:]:
            samples_arr = np.concatenate((samples_arr, data[item]))

        base_model = load_quantized_model(args.model)
        base_model = activations_rescaling(base_model, samples_arr,
                                           args.batch_size)
        base_model = normalize_separable_model(base_model)

        # Extract base name
        base_name = os.path.splitext(args.model)[0]

        if args.method == "adaround":
            calibrate_adaround_model(base_model, base_name, samples_arr,
                                     args.batch_size, args.epochs,
                                     args.learning_rate,
                                     args.include_activation)
        else:
            calibrate_bias_correction_model(base_model, base_name, samples_arr,
                                            args.batch_size)
