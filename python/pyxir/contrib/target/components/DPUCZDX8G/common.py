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

"""Module for registering common DPUCZDX8G functionality"""

import os
import logging

from pyxir.target import Target, DefaultOpSupportPass
from pyxir.graph import XGraph, XLayer
from pyxir.graph.layer.xlayer import defaultXLayer
from pyxir.graph.pattern import XGraphPatternMutator, XGraphPatternAnnotator
from pyxir.generator.tensorflow import XGraphTfGeneratorOptimizer
from pyxir.graph.optimization.optimizers import ExternalQOptimizer
from pyxir.graph.transformers.layout_transformation_pass import XGraphLayoutTransformationPass
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from pyxir.contrib.target.components.common import is_dpuczdx8g_vart_flow_enabled

logger = logging.getLogger('pyxir')


class OpSupportPass(DefaultOpSupportPass):

    def __init__(self, target):
        super().__init__(target)

    def __call__(self, xg: XGraph) -> None:
        """Call Pattern Annotator pass on XGraph before calling default op support functionality"""
        XGraphPatternAnnotator()(xg)
        super(OpSupportPass, self).__call__(xg)


def xgraph_dpu_op_support_annotator(xgraph: XGraph, target: Target, **kwargs) -> None:
    logger.info("xgraph op support pass in %r.", os.path.abspath(__file__))
    OpSupportPass(target)(xgraph)


def xgraph_dpu_optimizer(xgraph, target=None, **kwargs):
    logger.info("xgraph optimizer pass in %r.", os.path.abspath(__file__))
    XGraphPatternAnnotator()(xgraph)
    xgraph = XGraphPatternMutator()(xgraph)

    logger.debug("Call xgraph layout transformer")
    layout_transform_pass = XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)
    
    logger.debug("Call xgraph layout optimizer")
    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()
    return dpu_xgraph


def xgraph_dpu_quantizer(xgraph, inputs_func, **kwargs):
    logger.info("xgraph quantizer pass in %r.", os.path.abspath(__file__))
    quantizer = DECENTQuantizer(xgraph, inputs_func, compiler_target="xcompiler", **kwargs) \
        if is_dpuczdx8g_vart_flow_enabled() \
        else DECENTQuantizer(xgraph, inputs_func, **kwargs)
    q_xgraph = quantizer.quantize()
    return q_xgraph
