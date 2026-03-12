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
"""Functions to set/get Akida target version"""

__all__ = ["AkidaVersion", "get_akida_version", "set_akida_version"]

import os
from enum import Enum
from contextlib import contextmanager


class AkidaVersion(Enum):
    v1 = "v1"
    v2 = "v2"


TARGET_AKIDA_VERSION = "CNN2SNN_TARGET_AKIDA_VERSION"


def get_akida_version():
    """Get the target akida version for model conversion.

    Returns:
        AkidaVersion: the target akida version, by default ``AkidaVersion.v2``
    """
    ak_str_version = os.environ.get(TARGET_AKIDA_VERSION, "v2")
    ak_version = getattr(AkidaVersion, ak_str_version, None)
    if ak_version is None:
        raise ValueError(
            f"{TARGET_AKIDA_VERSION}={ak_str_version} must be one of {AkidaVersion._member_names_}")
    return ak_version


@contextmanager
def set_akida_version(version):
    """Select the target akida version for model conversion.

    Args:
        version (AkidaVersion): the target Akida version.
    """
    assert isinstance(version, AkidaVersion), "Version must be an AkidaVersion"

    _prev_state = os.environ.get(TARGET_AKIDA_VERSION, None)
    try:
        os.environ[TARGET_AKIDA_VERSION] = version.value
        yield
    finally:
        # Recover default value
        if _prev_state is not None:
            os.environ[TARGET_AKIDA_VERSION] = _prev_state
        else:
            os.environ.pop(TARGET_AKIDA_VERSION)
