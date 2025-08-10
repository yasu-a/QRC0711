from functools import reduce

import numpy as np

from core.fullstate import fullstate
from core.time_evol_solver import AbstractTimeEvolutionSolver, create_time_evol_solver
from experiment.estimator.base import AbstractReservoirEstimator
from model.axis import Axis
from model.dataset import Continuous, Discrete
from model.param import NVQRCParam
from model.physical_system import NVPhysicalSystem, EachSingleQubitSingleAxisObservable, \
    NVReservoirCollapseOperator, \
    AbstractResponsivePhysicalSystem
from model.state_array import QRCStateArray, AbstractState2DArray


class NVQRCEstimator(AbstractReservoirEstimator):
    @classmethod
    def _create_system(cls, param: NVQRCParam, seed: int) -> NVPhysicalSystem:
        return NVPhysicalSystem(
            n_qubit=param.n_qubits,
            j_mean=param.j_mean,
            j_std=param.j_std,
            j_axis=Axis.X,
            h_mean=param.h_mean,
            h_std=param.h_std,
            h_axis=Axis.Z,
            h_td_axis=Axis.X,
            seed=seed,
        )

    @classmethod
    def _create_solver(cls, *, param: NVQRCParam,
                       system: AbstractResponsivePhysicalSystem) -> AbstractTimeEvolutionSolver:
        return create_time_evol_solver(
            system=system,
            observable=reduce(
                lambda x, y: x + y,
                [
                    EachSingleQubitSingleAxisObservable(n_qubit=param.n_qubits, axis=axis)
                    for is_enabled, axis in
                    [(param.obs_x, Axis.X), (param.obs_y, Axis.Y), (param.obs_z, Axis.Z)]
                    if is_enabled
                ],
            ),
            collapse_operator=NVReservoirCollapseOperator(n_qubit=param.n_qubits,
                                                          gamma_z=param.gamma_z),
            init_psi=fullstate(",".join(["z+"] * param.n_qubits)),
        )

    def __init__(
            self,
            *,
            param: NVQRCParam,
            seed: int,
            n_washout: int,
            n_cpu: int,
    ) -> None:
        system = self._create_system(param=param, seed=seed)
        solver = self._create_solver(param=param, system=system)
        
        self._param = param

        super().__init__(
            system=system,
            solver=solver,
            seed=seed,
            n_washout=n_washout,
            n_cpu=n_cpu
        )

    def _get_time_evol_states(
            self,
            *,
            u_t: Continuous,
            t_seq: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None
    ) -> tuple[np.ndarray, QRCStateArray]:
        """
        NVQRC状態系列と出力時刻配列を計算する。

        Args:
            u_t (np.ndarray): 入力信号配列
            t_seq (np.ndarray): 時刻配列
            reset_state (bool, optional): 状態をリセットするかどうか。デフォルトはTrue。
            tqdm_title (str | None, optional): 進捗バーのタイトル。デフォルトはNone。

        Returns:
            tuple[np.ndarray, QRCStateArray]: 出力時刻配列とNVQRC状態系列。
        """
        return self._time_evol_series_computer.execute(
            solver=self._solver,
            n_mpx=self._param.n_mpx,
            u_t=u_t,
            t_arr=t_seq,
            reset_state=reset_state,
            tqdm_title=tqdm_title,
        )

    def _check_state_count(self, states: AbstractState2DArray) -> None:
        """NVQRC状態数の妥当性をチェックする。"""
        observable_count = sum([
            int(self._param.obs_x),
            int(self._param.obs_y),
            int(self._param.obs_z),
        ])
        expected_state_count = self._param.n_mpx * observable_count * self._param.n_qubits
        assert states.n_state == expected_state_count, (
            f"Invalid state count: expected {expected_state_count} states "
            f"(n_mpx={self._param.n_mpx} * observables={observable_count} "
            f"* n_qubits={self._param.n_qubits}), but got {states.n_state} states"
        )

    # TODO: _get_time_evol_statesのみをサービスとして切り出してテストを作り、一致を確認 非stiffなデータを使用しているから？
