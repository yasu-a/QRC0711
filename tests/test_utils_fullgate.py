from typing import Callable

import pytest
import qutip

from core.fullgate import fullgate

QObjProducer = Callable[[], qutip.Qobj]


@pytest.mark.parametrize(
    "n_qubit,fmt,fullgate_true",
    [
        (
                2,
                "X1",
                qutip.tensor(qutip.qeye(2), qutip.sigmax()),
        ),
        (
                2,
                "Y0",
                qutip.tensor(qutip.sigmay(), qutip.qeye(2)),
        ),
        (
                3,
                "X0,Y1,Z2",
                qutip.tensor(qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()),
        ),
        (
                4,
                "Z1",
                qutip.tensor(qutip.qeye(2), qutip.sigmaz(), qutip.qeye(2), qutip.qeye(2)),
        ),
    ]
)
def test_utils_fullgate(n_qubit: int, fmt: str, fullgate_true: qutip.Qobj):
    assert fullgate(n_qubit, fmt) == fullgate_true


@pytest.mark.parametrize(
    "n_qubit,fmt",
    [
        (2, "Z1,X0"),
        (2, "X0,Z0"),
    ]
)
def test_utils_fullgate_fmt_unordered_qubit_index(n_qubit: int, fmt: str):
    with pytest.raises(ValueError, match="all indexes must be unique and ordered"):
        fullgate(n_qubit, fmt)


@pytest.mark.parametrize(
    "n_qubit,fmt",
    [
        (2, "Z3"),
        (3, "X4"),
        (1, "Y2")
    ]
)
def test_utils_fullgate_fmt_invalid_qubit_index(n_qubit: int, fmt: str):
    with pytest.raises(ValueError, match="index is out of valid range"):
        fullgate(n_qubit, fmt)
