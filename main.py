from abc import ABC, abstractmethod
from functools import lru_cache, cache
from typing import Callable

import numpy as np
import qutip
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import fullgate
from utils.axis import Axis
from utils.dataset import train_test_split, Narma10, to_continuous_function
from utils.fullstate import fullstate
import os
import pickle

u_train, y_train, u_test, y_test = train_test_split(
    *Narma10.create_with_random_input(n=3000, seed=0),
    n_washout=500,
    test_ratio=0.3,
)


# plt.figure(figsize=(20, 5))
# plt.subplot(2, 1, 1)
# plt.plot(u_train, lw=0.3)
# plt.plot(y_train, lw=0.3)
# plt.subplot(2, 1, 2)
# plt.plot(u_test, lw=0.3)
# plt.plot(y_test, lw=0.3)
# plt.show()


class AbstractPhysicalSystem(ABC):
    @abstractmethod
    def create_hamiltonian(self, **kwargs):
        raise NotImplementedError()


class AbstractObservable(AbstractPhysicalSystem, ABC):
    pass


class AbstractCollapseOperator(AbstractPhysicalSystem, ABC):
    pass


class NVReservoirObservable(AbstractObservable):
    def __init__(self, *, n_qubit: int):
        self._n_qubit = n_qubit

    def create_hamiltonian(self):
        return [
            fullgate(self._n_qubit, f"X{i}")
            for i in range(self._n_qubit)
        ]


class NVReservoirCollapseOperator(AbstractCollapseOperator):
    def __init__(self, *, n_qubit: int, gamma_z: float):
        self._n_qubit = n_qubit
        self._gamma_z = gamma_z

    def create_hamiltonian(self):
        return [
            self._gamma_z ** .5 * fullgate(self._n_qubit, f"Z{i}")
            for i in range(self._n_qubit)
        ]


class FullInteraction(AbstractPhysicalSystem):
    # Jij * fullgate([[axis, i], [axis, j]] foreach i, j where i < j)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        assert coeff.ndim == 2 and coeff.shape[0] == coeff.shape[1], coeff.shape
        self._coeff = coeff
        self._axis = axis

    @classmethod
    def create_instance(cls, *, n_qubit: int, mean: float, std: float, axis: Axis):
        assert n_qubit >= 1
        values = np.random.normal(loc=mean, scale=std, size=(n_qubit * n_qubit - n_qubit) // 2)
        values *= np.random.choice([-1, +1], size=len(values))
        coeff = np.full((n_qubit, n_qubit), np.nan, np.float32)
        row_indices, col_indices = np.triu_indices(n_qubit, k=1)
        coeff[row_indices, col_indices] = values
        return cls(coeff, axis=axis)

    @property
    def n_qubit(self):
        return len(self._coeff)

    @cache
    def create_hamiltonian(self):
        return [
            self._coeff[i][j] * fullgate(self.n_qubit, f"{self._axis.name}{i},{self._axis.name}{j}")
            for i in range(self.n_qubit - 1)
            for j in range(i + 1, self.n_qubit)
        ]


class MagneticInteraction(AbstractPhysicalSystem):
    # hi * fullgate([[axis, i]] foreach i) * u_t(t)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        assert coeff.ndim == 1
        self._coeff = coeff
        self._axis = axis

    @classmethod
    def create_instance(cls, *, n_qubit: int, mean: float, std: float, axis: Axis):
        assert n_qubit >= 1
        values = np.random.normal(loc=mean, scale=std, size=n_qubit)
        values *= np.random.choice([-1, +1], size=len(values))
        coeff = np.array(values, dtype=np.float32)
        return cls(coeff, axis=axis)

    @property
    def n_qubit(self):
        return len(self._coeff)

    def create_hamiltonian(self, *, td_coeff: Callable[[float], float] = None):
        # Create list of Hamiltonians for each qubit, scaled by coefficients
        ham_lst = [
            self._coeff[i] * fullgate(self.n_qubit, f"{self._axis.name}{i}")
            for i in range(self.n_qubit)
        ]

        # If time-dependent coefficient provided, wrap Hamiltonians in tuples with td_coeff
        if td_coeff is not None:
            ham_lst = [
                [ham, td_coeff]
                for ham in ham_lst
            ]
        return ham_lst


class NVReservoirPhysicsSystem(AbstractPhysicalSystem):
    def __init__(
            self,
            *,
            n_qubit: int,
            j_mean: float,
            j_std: float,
            j_axis: Axis,
            h_mean: float,
            h_std: float,
            h_axis: Axis,
            h_td_axis: Axis,
    ):
        self._full_interaction = FullInteraction.create_instance(
            n_qubit=n_qubit,
            mean=j_mean,
            std=j_std,
            axis=j_axis,
        )
        self._magnetic_interaction = MagneticInteraction.create_instance(
            n_qubit=n_qubit,
            mean=h_mean,
            std=h_std,
            axis=h_axis,
        )
        self._time_dependent_magnetic_interaction = MagneticInteraction.create_instance(
            n_qubit=n_qubit,
            mean=h_mean,
            std=h_std,
            axis=h_td_axis,
        )

    def create_hamiltonian(self, *, td_coeff):
        ham_1 = self._full_interaction.create_hamiltonian()
        ham_2 = self._magnetic_interaction.create_hamiltonian()
        ham_3 = self._time_dependent_magnetic_interaction.create_hamiltonian(td_coeff=td_coeff)
        return ham_1 + ham_2 + ham_3


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


def cache_states(filename, compute_func):
    os.makedirs('./cache', exist_ok=True)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        result = compute_func()
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
        return result


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    # Simulation parameters
    N_QUBITS = 6
    GAMMA_Z = 0.001
    J_MEAN = 1.0
    J_STD = 0.1
    H_MEAN = 0.5
    H_STD = 0.1
    M = 500
    T = 150
    V = 2  # time-multiplexing index
    test_ratio = 0.3
    washout = 100

    t_arr = np.linspace(0, T, M)
    t_arr_div = np.linspace(0, T, M * V)
    u_arr = np.sin(t_arr_div) + np.random.RandomState(0).normal(0, 0.01, len(t_arr_div))
    y_arr = u_arr.copy()

    # Split into train/test with washout
    u_train, y_train, u_test, y_test = train_test_split(
        u_arr, y_arr, test_ratio=test_ratio, n_washout=washout
    )

    # Prepare solver
    system = NVReservoirPhysicsSystem(
        n_qubit=N_QUBITS,
        j_mean=J_MEAN,
        j_std=J_STD,
        j_axis=Axis.X,
        h_mean=H_MEAN,
        h_std=H_STD,
        h_axis=Axis.Z,
        h_td_axis=Axis.X,
    )
    solver = TimeEvolutionSolver(
        system=system,
        observable=NVReservoirObservable(n_qubit=N_QUBITS),
        collapse_operator=NVReservoirCollapseOperator(n_qubit=N_QUBITS, gamma_z=GAMMA_Z),
        init_psi=fullstate(",".join(["x+"] * N_QUBITS)),
    )

    # Helper to get states for a given u_arr and t_arr
    def get_states(u_arr, t_arr, solver):
        u_t = to_continuous_function(u_arr, t_step=np.diff(t_arr).mean())
        values = []
        for i in tqdm(range(len(t_arr) - 2)):
            t_begin, t_end = t_arr[i], t_arr[i + 1]
            t_div = np.linspace(t_begin, t_end, V + 1)[:-1]
            result = solver.forward(u_t, t_div)
            # flatten all qubit states at this time step
            state_vec = np.concatenate([result.expect(j) for j in range(result.n_expect)])
            values.append(state_vec)
        return np.array(values)

    # Get states for train and test (with cache)
    solver.reset_rho()
    t_arr_train = np.linspace(0, len(u_train) * (T / len(u_arr)), len(u_train))
    states_train = cache_states('./cache/state_train.pickle',
                                lambda: get_states(u_train, t_arr_train, solver))
    y_train_cut = y_train[1:len(states_train) + 1]  # align with state steps

    solver.reset_rho()
    t_arr_test = np.linspace(0, len(u_test) * (T / len(u_arr)), len(u_test))
    states_test = cache_states('./cache/state_test.pickle',
                               lambda: get_states(u_test, t_arr_test, solver))
    y_test_cut = y_test[1:len(states_test) + 1]

    # Linear regression
    reg = LinearRegression()
    reg.fit(states_train, y_train_cut)
    y_pred_train = reg.predict(states_train)
    y_pred_test = reg.predict(states_test)
    r2_train = r2_score(y_train_cut, y_pred_train)
    r2_test = r2_score(y_test_cut, y_pred_test)

    # Plot QRC state time series (values)
    plt.figure(figsize=(14, 6))
    for i in range(states_train.shape[1]):
        plt.plot(states_train[200:300, i], label=f'state {i}')
    plt.title('Train states (QRC values) time series')
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=False, width_ratios=[7, 2])
    # Train: time series
    axs[0, 0].plot(t_arr_train[1:len(states_train) + 1], y_train_cut, label='y_train (true)')
    axs[0, 0].plot(t_arr_train[1:len(states_train) + 1], y_pred_train, label='y_train (pred)',
                   linestyle='--')
    axs[0, 0].set_title(f'Train (R^2={r2_train:.3f})')
    axs[0, 0].set_ylabel('Output')
    axs[0, 0].legend()
    # Train: scatter
    axs[0, 1].scatter(y_train_cut, y_pred_train, alpha=0.5)
    minv = min(y_train_cut.min(), y_pred_train.min())
    maxv = max(y_train_cut.max(), y_pred_train.max())
    axs[0, 1].plot([minv, maxv], [minv, maxv], 'k--', label='y=true')
    axs[0, 1].set_xlabel('y_true')
    axs[0, 1].set_ylabel('y_pred')
    axs[0, 1].set_title(f'Train: y_true vs y_pred\nR²={r2_train:.3f}')
    axs[0, 1].legend()
    axs[0, 1].set_aspect('equal', adjustable='box')
    # Test: time series
    axs[1, 0].plot(t_arr_test[1:len(states_test) + 1], y_test_cut, label='y_test (true)')
    axs[1, 0].plot(t_arr_test[1:len(states_test) + 1], y_pred_test, label='y_test (pred)',
                   linestyle='--')
    axs[1, 0].set_title(f'Test (R^2={r2_test:.3f})')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Output')
    axs[1, 0].legend()
    # Test: scatter
    axs[1, 1].scatter(y_test_cut, y_pred_test, alpha=0.5)
    minv = min(y_test_cut.min(), y_pred_test.min())
    maxv = max(y_test_cut.max(), y_pred_test.max())
    axs[1, 1].plot([minv, maxv], [minv, maxv], 'k--', label='y=true')
    axs[1, 1].set_xlabel('y_true')
    axs[1, 1].set_ylabel('y_pred')
    axs[1, 1].set_title(f'Test: y_true vs y_pred\nR²={r2_test:.3f}')
    axs[1, 1].legend()
    axs[1, 1].set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
