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
__all__ = ["register_onnx_converter_target", "map_node_to_converter"]

from .base_converter import OnnxConverter

_ONNX_CONVERTERS = {}


def register_onnx_converter_target(root_op_type):
    """Register the current class as a target for converter modules.

    This decorator injects the decorated class into the _ONNX_CONVERTERS dictionary, so that
    it is registered to as converter target.

    Args:
        root_op_type (str): string that represents the base operation type
            which will map the decorator's target class.

    Returns:
        Callable: a decorator that registers the decorated class
    """
    def _register_target(target, arg):
        """Register arg with target name.

        Args:
            target (str): the target name to register
            arg (cls): the current class to register
        """
        if not issubclass(arg, OnnxConverter):
            raise ValueError("Can only register OnnxCoverter class objects.")

        if target in _ONNX_CONVERTERS:
            raise ValueError(f"{target} has already been registered to {_ONNX_CONVERTERS[target]}.")
        _ONNX_CONVERTERS.update({target: arg})

    def decorator(arg):
        assert isinstance(root_op_type, str)
        _register_target(root_op_type, arg)
        return arg
    return decorator


def map_node_to_converter(node, model):
    """Map node into its converter version.

    Args:
        node (NodeProto): the node to map.
        model (ModelProto): the model to initialize the converter.

    Raises:
        RuntimeError: when node is not register as a valid operation.

    Returns:
        OnnxConverter: the converter.
    """
    # Sort converters by the length of the operation type in descending order
    # to prioritize more specific matches (longer prefixes).
    sorted_converters = sorted(_ONNX_CONVERTERS.items(), key=lambda x: len(x[0]), reverse=True)

    for root_op_type, converter in sorted_converters:
        if node.op_type.startswith(root_op_type):
            return converter(node, model)
    raise RuntimeError(f"Unrecognized {node.op_type} operation. Not register yet!")
