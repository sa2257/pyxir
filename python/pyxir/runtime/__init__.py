# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Module for executing XGraphs """

import warnings

from .rt_manager import RtManager
from .runtime_factory import RuntimeFactory

rt_manager = RtManager()
runtime_factory = RuntimeFactory()


try:
    # Register if we can import tensorflow
    from .tensorflow.runtime_tf import RuntimeTF, X_2_TF

    rt_manager.register_rt('cpu-tf', RuntimeTF, X_2_TF)
except Exception as e:
    warnings.warn("Could not load `cpu-tf` runtime because of error: {}"
                  .format(e))

try:
    # Register if we can import numpy
    from .numpy.runtime_np import RuntimeNP, X_2_NP

    rt_manager.register_rt('cpu-np', RuntimeNP, X_2_NP)
except Exception as e:
    warnings.warn("Could not load `cpu-np` runtime because of error: {}"
                  .format(e))
