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
"""Helper to update V1 models act_step if too high to be mappable"""

import warnings
import numpy as np
import akida

# act_step should fit in 16bit
ACT_STEP_HW_LIMIT = (1 << 16) - 1
THRESHOLD_HW_LIMIT = (1 << 19) - 1


def fix_v1_activation_variables(model):
    """ Helper to equalize the Akida model act_step if those are not mappable.

    If the act_step of a layer is higher the value allowed in hardware, this method will update
    this value and modify the model to adapt it to the limit. To do that, it will compute the ratio
    by which the act_step should be divide to fit in the HW limit should be divide to fit in the
    HW limit and then divide the weights of the next layer by the next ratio to keep the
    equivalence.

    Args:
        model (`akida.Model`): an Akida model.

    """
    # This operation is compatible only for akida v1 target
    if model.ip_version != akida.core.IpVersion.v1:
        return
    # Equalize model act_steps
    _equalize_act_step(model)

    # Saturate thresholds
    _saturate_threshold(model)


def _equalize_act_step(model):
    for id, layer in enumerate(model.layers[:-1]):
        if "act_step" in layer.variables.names:
            act_step = layer.variables["act_step"]
            if (act_step > ACT_STEP_HW_LIMIT).any():
                original_act_step = act_step.copy()
                # Compute the ratio by which the act_step should be divide to fit in the
                # HW requirement
                ratio = np.where(act_step > ACT_STEP_HW_LIMIT, act_step / ACT_STEP_HW_LIMIT, 1)
                # Update the non valid act_steps
                act_step[act_step > ACT_STEP_HW_LIMIT] = ACT_STEP_HW_LIMIT
                layer.variables["act_step"] = act_step
                # If the layer has an activation, the threshold should also be updated
                # since it holds a part of the act_step. To do so we assume that the Akida
                # model has been converted through the cnn2snn API
                if layer.parameters.activation:
                    act_bits = layer.parameters.act_bits
                    # Remove from the threshold the effect of the original act_step,
                    # then update with the new one
                    threshold_effect = np.round(0.5 * (act_step - original_act_step))
                    threshold_effect /= 2 ** (act_bits - 4)
                    layer.variables["threshold"] += threshold_effect.astype(np.int32)
                # get the next layer weights
                next_weights = model.get_layer(id + 1).variables["weights"]
                # divide the next layer weights by the ratio to keep the equivalence
                next_weights = next_weights // ratio[np.newaxis, np.newaxis, :, np.newaxis]
                model.get_layer(id + 1).variables["weights"] = next_weights.astype(np.int8)


def _saturate_threshold(model):
    for layer in model.layers[:-1]:
        if "threshold" in layer.variables.names:
            threshold = layer.variables["threshold"]
            # Saturate the threshold value to the max threshold value supported by the HW
            # Note: that this might slightly affect the global model accuracy in comparison
            # with its quantized version.
            if np.any(threshold > THRESHOLD_HW_LIMIT):
                threshold[threshold > THRESHOLD_HW_LIMIT] = THRESHOLD_HW_LIMIT
                warnings.warn(f"The {layer.name} layer holds very high threshold values which are "
                              "not compatible with the hardware. Those are clipped to the maximum "
                              f"supported value {THRESHOLD_HW_LIMIT}. Continuing execution.")
                layer.variables["threshold"] = threshold
