import numpy as np
import pytest
import qutip

from core.fullgate import fullgate
from core.fullstate import fullstate


@pytest.mark.parametrize(
    "initial_state, hamiltonian, observable, expected_fn",
    [
        ("Z+", "X0", "Y0", lambda t: -np.sin(t)),
        ("Z+", "Y0", "X0", lambda t: np.sin(t)),
        ("Z+", "X0", "Z0", lambda t: np.cos(t)),
        ("X+", "Z0", "X0", lambda t: np.cos(t)),
        ("X+", "Z0", "Y0", lambda t: np.sin(t)),
        ("Y+", "X0", "Z0", lambda t: np.sin(t)),
        ("Z-", "Y0", "X0", lambda t: -np.sin(t)),
        ("Y-", "Z0", "X0", lambda t: np.sin(t)),
        ("Y-", "X0", "Y0", lambda t: -np.cos(t)),
        ("X-", "Y0", "Z0", lambda t: np.sin(t)),
        ("X-", "Z0", "Y0", lambda t: -np.sin(t)),
        ("Y+", "Z0", "X0", lambda t: -np.sin(t)),
        ("Y-", "Z0", "Y0", lambda t: -np.cos(t)),
        ("Z+", "Z0", "X0", lambda t: 0 * t),
        ("Z-", "Z0", "Y0", lambda t: 0 * t),
        ("X+", "X0", "Y0", lambda t: 0 * t),
        ("X-", "X0", "Z0", lambda t: 0 * t),
        ("Y+", "Y0", "Z0", lambda t: 0 * t),
        ("Y-", "Y0", "X0", lambda t: 0 * t),
    ],
)
def test_fullstate_with_rotation(initial_state, hamiltonian, observable, expected_fn):
    psi = fullstate(initial_state)
    ham = fullgate(1, hamiltonian) / 2
    obs = fullgate(1, observable)
    t = np.linspace(0, 10, 100)
    result = qutip.mesolve(ham, psi, tlist=t, e_ops=[obs])
    np.testing.assert_array_almost_equal(result.expect[0], expected_fn(t), decimal=5)
