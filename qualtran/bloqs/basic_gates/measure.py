#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from cirq.ops import Operation, QubitManager
import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    DecomposeTypeError,
    Register,
    Signature,
    SoquetT,
    QBit,
    Side,
)
from qualtran.cirq_interop import CirqQuregT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Text, TextBox, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT

@frozen
class Measure(Bloq):
    r"""Measurement operation which consumes a single qubit.

    Currently does not support actual measurements and is mostly a stub.
    Registers:
        r: The register
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(registers=[Register('r', QBit(), side=Side.LEFT)])
    
    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

