import numpy as np
import qutip
from numpy.typing import ArrayLike

from physical_system import AbstractPhysicalSystem, AbstractObservable, AbstractCollapseOperator
from utils.app_logging import create_logger


class TimeEvolutionSolver:
    _logger = create_logger()

    def __init__(
            self,
            *,
            system: AbstractPhysicalSystem,
            observable: AbstractObservable,
            collapse_operator: AbstractCollapseOperator,
            init_psi: qutip.Qobj,
    ):
        self._system = system
        self._observable = observable
        self._collapse_operator = collapse_operator

        # Create initial density matrix from pure state
        # noinspection PyTypeChecker
        self._init_rho = init_psi * init_psi.dag()
        self._rho = self._init_rho

        # Flag to track if imaginary value warning has been shown
        self._large_expect_imag_warned = False

    class ForwardResult:
        def __init__(self, *, expect, final_state):
            # Store expectation values and final state
            self._expect = expect  # List of expectation values for observables 
            self._final_state = final_state  # Final density matrix

        @property
        def n_expect(self) -> int:
            # Number of observables we got expectations for
            return len(self._expect)

        def expect(self, i) -> np.ndarray:
            # Get an array of i-th expectation value
            return self._expect[i]

        @property
        def final_rho(self) -> qutip.Qobj:
            # Get final density matrix state
            return self._final_state

    def reset_rho(self):
        # Reset density matrix to initial state
        self._rho = self._init_rho

    def _warn_large_expect_imag(self, imag_max: float):
        # Show warning once if expectation values have large imaginary parts
        if not self._large_expect_imag_warned:
            self._logger.warning(
                f"Expectation values retuned by mesolve contains large imaginary part: "
                f"{imag_max} max. This warning will be shown only once for this instance."
            )
            self._large_expect_imag_warned = True

    def _coerce_expect(self, expect: list[ArrayLike]):
        # Handle imaginary parts in expectation values
        new_expect = []
        for e_arr in expect:
            if np.iscomplexobj(e_arr):
                imag_max = np.abs(e_arr.astype(complex).imag).max()
                if imag_max != 0:
                    self._warn_large_expect_imag(imag_max)
                e_arr = e_arr.real
            new_expect.append(e_arr)
        return new_expect

    def forward(self, u_t, t_arr) -> ForwardResult:
        # Calculate average time step
        avg_step = np.mean(np.diff(t_arr))
        self._logger.debug(
            f"Forward: "
            f"t_arr={len(t_arr)} points from {t_arr[0]} to {t_arr[-1]}, "
            f"avg_step={avg_step:.6f}"
        )

        # Solve quantum master equation
        result = qutip.mesolve(
            H=self._system.create_hamiltonian(td_coeff=u_t),
            rho0=self._rho,
            tlist=t_arr,
            c_ops=self._collapse_operator.create_hamiltonian(),
            e_ops=self._observable.create_hamiltonian(),
            options=dict(store_final_state=True)
        )

        # Package results and update system state
        result = self.ForwardResult(
            expect=self._coerce_expect(result.expect),
            final_state=result.final_state,  # 期待値に虚部が含まれることがある
        )
        self._rho = result.final_rho
        return result
