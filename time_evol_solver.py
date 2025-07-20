import numpy as np
import qutip

from physical_system import AbstractPhysicalSystem, AbstractObservable, AbstractCollapseOperator


class TimeEvolutionSolver:
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

        # noinspection PyTypeChecker
        self._init_rho = init_psi * init_psi.dag()
        self._rho = self._init_rho

    class ForwardResult:
        def __init__(self, expect, final_state):
            self._expect = expect
            self._final_state = final_state

        @property
        def n_expect(self) -> int:
            return len(self._expect)

        def expect(self, i) -> np.ndarray:
            return self._expect[i]

        @property
        def final_rho(self) -> qutip.Qobj:
            return self._final_state

    def reset_rho(self):
        self._rho = self._init_rho

    def forward(self, u_t, t_arr) -> ForwardResult:
        result = qutip.mesolve(
            H=self._system.create_hamiltonian(td_coeff=u_t),
            rho0=self._rho,
            tlist=t_arr,
            c_ops=self._collapse_operator.create_hamiltonian(),
            e_ops=self._observable.create_hamiltonian(),
            options=dict(store_final_state=True)
        )
        result = self.ForwardResult(result.expect, result.final_state)
        self._rho = result.final_rho
        return result
