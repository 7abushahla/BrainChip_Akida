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
"""Helper functions to convert to akida models, models quantized with quantizeml API.
"""
from .model_generator import generate_model
from .block_converters_generator import generate_block_converters
from .dense import *
from .convolution import *
from .input_conv import *
from .input_quantizer import *
from .separable_convolution import *
from .conv2d_transpose import *
from .depthwise_conv2d import *
from .depthwise_conv2d_transpose import *
from .add import *
from .concatenate import *
from .dequantizer import *
from .buffer_temp_conv import *
from .depthwise_buffer_temp_conv import *
from .stateful_recurrent import *
