import warnings
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import qutip
from numpy.typing import ArrayLike
from qutip import Qobj

from core.app_logging import create_logger
from core.fullgate import fullgate
from core.fullstate import fullstate
from core.seed_or_rng import check_seed_or_rng_and_get_rng
from model.physical_system import AbstractCollapseOperator, \
    AbstractResponsivePhysicalSystem

_N_QUBIT_MAPPING = {2 ** i: i for i in range(64)}


@dataclass(slots=True)
class StepResult:
    t: float
    rho: Qobj  # shape: (2^L, 2^L)

    def __post_init__(self):
        assert isinstance(self.t, float)
        assert isinstance(self.rho, Qobj)
        assert self.rho.type == "oper", self.rho.type
        if self.rho.shape[0] not in _N_QUBIT_MAPPING:
            raise ValueError(
                f"rho.shape[0] is not a power of 2: {self.rho.shape[0]}"
            )
        if self.rho.shape[0] != self.rho.shape[1]:
            raise ValueError(
                f"rho must be square matrix: {self.rho.shape}"
            )

    @property
    def n_qubit(self) -> int:
        return _N_QUBIT_MAPPING[self.rho.shape[0]]

    def expect(self, obs: Qobj) -> float:
        e = qutip.expect(obs, self.rho)
        if e.imag != 0:
            warnings.warn(
                f"Expectation value has large imaginary part: {e.imag:.6g}")
        return e.real

    def variance(self, obs: Qobj) -> float:
        e = qutip.variance(obs, self.rho)
        if e.imag != 0:
            warnings.warn(f"Variance has large imaginary part: {e.imag:.6g}")
        return e.real

    def std(self, obs: Qobj) -> float:
        return np.sqrt(self.variance(obs))

    def sample(self, obs: Qobj, n_samples: int, rng: np.random.RandomState) \
            -> np.ndarray:  # shape: (n_samples,)
        # オブザーバブルの固有値と固有ベクトルを計算
        eigvals, eigvecs = obs.eigenstates()

        # 固有値に虚数部がないかチェック
        if not np.any(np.isclose(eigvals.imag, 0)):
            warnings.warn(
                f"Observable has eigenvalues with non-zero imaginary parts. "
                f"Max imag part: {np.max(np.abs(eigvals.imag)):.6g}"
            )

        # 各固有値が観測される確率を計算
        probabilities = np.array([
            (eigvec.dag() * self.rho * eigvec).real
            for eigvec in eigvecs
        ])

        # たまに確率が負になるので0にする
        probabilities = np.where(probabilities <= 0, 0, probabilities)

        # 確率の正規化チェック (浮動小数点誤差のため)
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0):
            warnings.warn(
                f"Probabilities do not sum to 1.0, but to {prob_sum:.6g}. Normalizing.")
        probabilities /= prob_sum

        # 観測値をサンプリング
        samples = rng.choice(a=eigvals.real, p=probabilities, size=n_samples)

        return samples

    def sample_with_post_rho(self, obs: Qobj, n_samples: int, rng: np.random.RandomState) \
            -> tuple[np.ndarray, list[Qobj]]:  # shape: (n_samples,) for each item in tuple
        # オブザーバブルの固有値と固有ベクトルを計算
        eigvals, eigvecs = obs.eigenstates()

        # 固有値に虚数部がないかチェック
        if not np.any(np.isclose(eigvals.imag, 0)):
            warnings.warn(
                f"Observable has eigenvalues with non-zero imaginary parts. "
                f"Max imag part: {np.max(np.abs(eigvals.imag)):.6g}"
            )

        # 各固有値が観測される確率を計算
        probabilities = np.array([
            (eigvec.dag() * self.rho * eigvec).real
            for eigvec in eigvecs
        ])

        # 確率の正規化チェック (浮動小数点誤差のため)
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0):
            warnings.warn(
                f"Probabilities do not sum to 1.0, but to {prob_sum:.6g}. Normalizing.")
            probabilities /= prob_sum

        # 観測値をサンプリング
        samples_idx = rng.choice(
            a=len(eigvals), p=probabilities, size=n_samples)
        samples = eigvals.real[samples_idx]

        # サンプリング後の状態（射影後のrho）を計算
        post_rho_list: list[Qobj] = []
        for idx in samples_idx:
            eigvec = eigvecs[idx]
            proj = eigvec * eigvec.dag()
            post_rho = proj  # 射影測定後の密度行列
            post_rho_list.append(post_rho)

        return samples, post_rho_list


if __name__ == '__main__':
    def _test_step_result():
        # この関数 _test_step_result は、StepResult クラスのサンプリング機能と期待値・標準偏差計算の正しさをテストするための関数です。
        # 具体的には、ランダムな重みで superposition 状態を作り、その状態に対して Z オブザーバブルの期待値・標準偏差を計算し、
        # サンプリングによる推定値と理論値が一致することを 1000 回繰り返して検証します。
        # 最後に、計算値とサンプル値を散布図で可視化し、理想的な一致（y=x）をプロットします。

        import random
        import matplotlib.pyplot as plt

        mean_calc, std_calc, mean_sample, std_sample = [], [], [], []
        for _ in range(1000):
            r = random.random()
            psi = fullstate("z+") * r + fullstate("z-") * (1 - r)
            psi = psi.unit()

            obs = fullgate(1, "Z0")

            rho = psi * psi.dag()  # 密度行列に変換
            res = StepResult(t=0.0, rho=rho)
            rng = check_seed_or_rng_and_get_rng(seed=0)
            samples = res.sample(obs, 10000, rng)
            mean_sample.append(samples.mean())
            std_sample.append(samples.std())

            mean_calc.append(res.expect(obs))
            std_calc.append(res.std(obs))

        plt.figure()
        plt.scatter(mean_calc, mean_sample, label="calc")
        plt.scatter(std_calc, std_sample, label="sample")
        plt.plot([-1, 1], [-1, 1], color="black", linestyle="--")
        plt.legend()
        plt.show()


    _test_step_result()


class TimeEvolutionResult:
    @classmethod
    def _validate(cls, lst: Sequence[StepResult]) -> None:
        if len(lst) == 0:
            raise ValueError("step_results is empty")
        if not all(isinstance(item, StepResult) for item in lst):
            raise ValueError("step_results contains non-StepResult items")
        if len(lst) >= 2:
            if not all(lst[0].n_qubit == item.n_qubit for item in lst):
                raise ValueError(
                    "step_results contains StepResult with different n_qubit")

    def __init__(self, *, step_results: Sequence[StepResult]):
        """
        時間発展の結果をStepResultのリストとして保持する。

        Args:
            step_results (list[StepResult]): 各時刻でのStepResultのリスト
        """
        self._validate(step_results)
        self._step_results = step_results

    @property
    def n_qubit(self) -> int:
        return self._step_results[0].n_qubit

    def __len__(self) -> int:
        return len(self._step_results)

    @property
    def times(self) -> np.ndarray:
        return np.array([step.t for step in self._step_results])

    def expect(self, obs: Qobj) -> np.ndarray:
        return np.array([step.expect(obs) for step in self._step_results])

    def variance(self, obs: Qobj) -> np.ndarray:
        return np.array([step.variance(obs) for step in self._step_results])

    def std(self, obs: Qobj) -> np.ndarray:
        return np.array([step.std(obs) for step in self._step_results])

    @property
    def final_rho(self) -> Qobj:
        final_step = self._step_results[-1]
        return final_step.rho

    def sample_measurements(self, obs: Qobj, *, n_samples: int, rng: np.random.RandomState) \
            -> np.ndarray:  # shape: (n_obs,)
        samples = self._step_results[-1].sample(obs=obs, n_samples=n_samples, rng=rng)
        return samples


class AbstractTimeEvolutionSolver(ABC):
    def __init__(
            self,
            *,
            system: AbstractResponsivePhysicalSystem,
            collapse_operator: AbstractCollapseOperator | None = None,
            init_psi: qutip.Qobj,
    ):
        self._system = system
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
    def forward(self, u_t: ArrayLike, t_seq: ArrayLike, *, result_index: ArrayLike | None = None,
                pbar_title: str | None = None) \
            -> TimeEvolutionResult:
        """
        Solve the quantum master equation and obtain the time series of expectation values for observables.

        Args:
            u_t (ArrayLike): Time-dependent coefficients for the Hamiltonian.
            t_seq (ArrayLike): Array of time points to evaluate.
            result_index (ArrayLike | None, optional): Indices of time points to include in result. 
                If None, all time points are included.
            pbar_title (str | None, optional): Title for the progress bar (optional).

        Returns:
            TimeEvolutionResult: Result containing the expectation value series and the final quantum state.
        """
        raise NotImplementedError()


class QutipMESolveTimeEvolutionSolver(AbstractTimeEvolutionSolver):
    _logger = create_logger()

    def __init__(
            self,
            *,
            system: AbstractResponsivePhysicalSystem,
            collapse_operator: AbstractCollapseOperator | None = None,
            init_psi: qutip.Qobj,
    ):
        super().__init__(
            system=system,
            collapse_operator=collapse_operator,
            init_psi=init_psi,
        )

        # Create initial density matrix from pure state
        # noinspection PyTypeChecker
        self._init_rho = self._init_psi * self._init_psi.dag()

        self._current_time: float = 0.0
        self._current_rho = self._init_rho

        # Flag to track if imaginary value warning has been shown
        self._large_expect_imag_warned = False

    def reset_state(self) -> None:
        # Reset density matrix to initial state
        self._current_time = 0.0
        self._current_rho = self._init_rho

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

    def forward(self, u_t: ArrayLike, t_seq: ArrayLike, *, result_index: ArrayLike | None = None,
                pbar_title: str | None = None) \
            -> TimeEvolutionResult:
        if t_seq[0] != self._current_time:
            raise ValueError(
                f"t_seq[0] is not equal to current time: "
                f"({t_seq[0]=}) != (current_time={self._current_time})"
            )

        # Calculate average time step
        avg_step = np.mean(np.diff(t_seq))
        self._logger.debug(
            f"Forward: "
            f"t_arr={len(t_seq)} points from {t_seq[0]} to {t_seq[-1]}, "
            f"avg_step={avg_step:.6f}"
        )

        # Define unified callback to store states
        stored_step_results: list[StepResult] = []
        _index = 0
        _index_to_store = set(map(int, result_index)) if result_index is not None else None

        def store_state_callback(t, rho):
            nonlocal _index
            if _index_to_store is None or _index in _index_to_store:
                stored_step_results.append(StepResult(t=float(t), rho=rho))
            _index += 1

        # Set options (no callback option in qutip 5.2)
        options: dict[str, Any] = {
            "store_final_state": True,
        }
        if pbar_title is not None:
            options["progress_bar"] = "tqdm"
            options["progress_kwargs"] = {"desc": pbar_title}

        # Set up collapse operator
        if self._collapse_operator is None:
            c_ops = None
        else:
            c_ops = self._collapse_operator.create_hamiltonian()

        # Solve the master equation
        # Note: Using e_ops with function to simulate callback functionality for qutip 5.2
        qutip_result = qutip.mesolve(
            H=self._system.create_hamiltonian(u_t=u_t),
            rho0=self._current_rho,
            tlist=t_seq,
            c_ops=c_ops,
            e_ops=[store_state_callback],
            options=options,
        )

        # Create result and update system state
        time_evol_result = TimeEvolutionResult(step_results=stored_step_results)

        # Verify times match expected indices (if result_index was specified)
        if result_index is not None:
            expected_times = t_seq[result_index]
            actual_times = time_evol_result.times
            assert np.isclose(actual_times, expected_times, atol=1e-12).all(), \
                f"Times mismatch: expected {expected_times}, got {actual_times}"

        self._current_rho = qutip_result.final_state
        self._current_time = float(t_seq[-1])
        return time_evol_result


_SOLVER_MAPPING: dict[str, type[AbstractTimeEvolutionSolver]] = {
    "qutip-mesolve": QutipMESolveTimeEvolutionSolver,
}


def create_time_evol_solver(
        *,
        system: AbstractResponsivePhysicalSystem,
        collapse_operator: AbstractCollapseOperator | None = None,
        init_psi: qutip.Qobj,
        backend: str = "qutip-mesolve",
) -> AbstractTimeEvolutionSolver:
    return _SOLVER_MAPPING[backend](
        system=system,
        collapse_operator=collapse_operator,
        init_psi=init_psi,
    )
