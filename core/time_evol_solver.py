from abc import abstractmethod, ABC

import numpy as np
import qutip
from numpy.typing import ArrayLike

from core.app_logging import create_logger
from model.physical_system import AbstractPhysicalSystem, AbstractObservable, \
    AbstractCollapseOperator


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


class AbstractTimeEvolutionSolver(ABC):
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
        self._init_psi = init_psi

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the solver state back to initial conditions.
        
        This resets any internal state like density matrices back to their initial values,
        allowing the solver to be reused for multiple forward passes from the same starting point.
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, u_t: ArrayLike, t_arr: ArrayLike) -> ForwardResult:
        """Solve quantum master equation and get expectation values of observables.

        Args:
            u_t: Time-dependent coefficients for the Hamiltonian
            t_arr: Array of time points to evaluate at

        Returns:
            ForwardResult containing expectation values and final quantum state
        """
        raise NotImplementedError()


class QutipMESolveTimeEvolutionSolver(AbstractTimeEvolutionSolver):
    _logger = create_logger()

    def __init__(
            self,
            *,
            system: AbstractPhysicalSystem,
            observable: AbstractObservable,
            collapse_operator: AbstractCollapseOperator,
            init_psi: qutip.Qobj,
    ):
        super().__init__(
            system=system,
            observable=observable,
            collapse_operator=collapse_operator,
            init_psi=init_psi,
        )

        # Create initial density matrix from pure state
        # noinspection PyTypeChecker
        self._init_rho = self._init_psi * self._init_psi.dag()
        self._rho = self._init_rho

        # Flag to track if imaginary value warning has been shown
        self._large_expect_imag_warned = False

    def reset_state(self) -> None:
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

    def forward(self, u_t: ArrayLike, t_arr: ArrayLike) -> ForwardResult:
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
            options=dict(
                store_final_state=True,
                # progress_bar="tqdm",
            ),
        )

        # Package results and update system state
        result = ForwardResult(
            expect=self._coerce_expect(result.expect),
            final_state=result.final_state,  # 期待値に虚部が含まれることがある
        )
        self._rho = result.final_rho
        return result


_SOLVER_MAPPING: dict[str, type[AbstractTimeEvolutionSolver]] = {
    "qutip-mesolve": QutipMESolveTimeEvolutionSolver,
}


def create_time_evol_solver(
        *,
        system: AbstractPhysicalSystem,
        observable: AbstractObservable,
        collapse_operator: AbstractCollapseOperator,
        init_psi: qutip.Qobj,
        backend: str = "qutip-mesolve",
) -> AbstractTimeEvolutionSolver:
    return _SOLVER_MAPPING[backend](
        system=system,
        observable=observable,
        collapse_operator=collapse_operator,
        init_psi=init_psi,
    )
