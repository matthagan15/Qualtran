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
from qualtran import BloqBuilder, Register
from qualtran.bloqs.arithmetic import (
    MultiplyTwoReals,
    Product,
    ScaleIntByReal,
    Square,
    SquareRealNumber,
    SumOfSquares,
)
from qualtran.testing import execute_notebook


def _make_square():
    from qualtran.bloqs.arithmetic import Square

    return Square(bitsize=8)


def _make_sum_of_squares():
    from qualtran.bloqs.arithmetic import SumOfSquares

    return SumOfSquares(bitsize=8, k=4)


def _make_product():
    from qualtran.bloqs.arithmetic import Product

    return Product(a_bitsize=4, b_bitsize=6)


def _make_scale_int_by_real():
    from qualtran.bloqs.arithmetic import ScaleIntByReal

    return ScaleIntByReal(r_bitsize=8, i_bitsize=12)


def _make_multiply_two_reals():
    from qualtran.bloqs.arithmetic import MultiplyTwoReals

    return MultiplyTwoReals(bitsize=10)


def _make_square_real_number():
    from qualtran.bloqs.arithmetic import SquareRealNumber

    return SquareRealNumber(bitsize=10)


def test_square():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q0, q1 = bb.add(Square(bitsize), a=q0)
    cbloq = bb.finalize(a=q0, result=q1)
    cbloq.t_complexity()


def test_sum_of_squares():
    bb = BloqBuilder()
    bitsize = 4
    k = 3
    inp = bb.add_register(Register("input", bitsize=bitsize, shape=(k,)))
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp)
    cbloq = bb.finalize(input=inp, result=out)
    assert SumOfSquares(bitsize, k).signature[1].bitsize == 2 * bitsize + 2
    cbloq.t_complexity()


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_scale_int_by_real():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 8)
    q0, q1, q2 = bb.add(ScaleIntByReal(15, 8), real_in=q0, int_in=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_multiply_two_reals():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(MultiplyTwoReals(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_square_real_number():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(SquareRealNumber(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)


def test_arithmetic_notebook():
    execute_notebook('arithmetic')