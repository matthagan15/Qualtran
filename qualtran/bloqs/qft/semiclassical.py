#  Copyright 2023 Google LLC
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
from typing import Iterator

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import QUInt, Signature, Bloq, Register, Side
from qualtran._infra.composite_bloq import BloqBuilder
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
from qualtran.symbolics import SymbolicInt

@attrs.frozen
class SemiclassicalQFT(Bloq):
    r"""Variant of QFT that measures registers to replace coherent rotations w/ classical control.

    In some quantum algorithms, such as Shor's factoring algorithm or Quantum Phase Estimation, the quantum fourier transform is used right before measurement. The semiclassical QFT relies on the insight that the coherently controlled rotations in the QFT can be replaced with classically controlled rotations if the qubits are measured in an appropriate order. 

    Since the number of rotations is now a nondeterministic quantity, Qualtran 
    provides a worst case analysis of the number of rotations that may be needed. This can still lead to a constant factor improvement in the number 
    of T gates needed. 
    Args:
        bitsize: Size of the input register to apply the Bloq to.

    Registers:
        q: The register to be consumed by the Bloq.

    References:
        
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            registers=[Register(name='q', dtype=QUInt(self.bitsize), side=Side.LEFT)]
        )
    

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: Soquet | NDArray[attrs.Any, dtype]) -> attrs.Dict[str, Soquet | NDArray[attrs.Any, dtype]]:
        return super().build_composite_bloq(bb, **soqs)