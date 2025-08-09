from functools import reduce
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from core.fullstate import fullstate
from core.time_evol_solver import create_time_evol_solver
from experiment.estimator.base import AbstractEstimator
from experiment.estimator.dto import StateComputationResult
from model.axis import Axis
from model.dataset import Continuous, Discrete, Dataset
from model.param import NVQRCParam
from model.physical_system import NVReservoirPhysicsSystem, AbstractPhysicalObject, \
    EachSingleQubitSingleAxisObservable, NVReservoirCollapseOperator
from model.prediction_result import PredictionResult
from model.state_array import QRCStateArray, AbstractState2DArray
from service.compute_time_evol import get_compute_time_evol_state_series_service


class NVQRCEstimator(AbstractEstimator):
    @classmethod
    def _create_system(cls, param: NVQRCParam, seed: int):
        return NVReservoirPhysicsSystem(
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
    def _create_solver(cls, *, param: NVQRCParam, system: AbstractPhysicalObject):
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
    ):
        assert n_cpu == -1 or n_cpu >= 1, f"invalid {n_cpu=}"

        self._param = param
        self._seed = seed
        self._n_washout = n_washout
        self._n_cpu = n_cpu

        self._lr_model = LinearRegression()
        self._system = self._create_system(param=param, seed=seed)
        self._solver = self._create_solver(param=param, system=self._system)
        self._time_evol_series_computer = get_compute_time_evol_state_series_service()

    def _get_time_evol_states(
            self,
            *,
            n_mpx: int,
            u_t: Continuous,
            t_seq: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None
    ) -> tuple[np.ndarray, QRCStateArray]:
        """
        QRC状態系列と出力時刻配列を計算する。

        Args:
            n_mpx (int): 各時刻区間を分割する数（マルチプレクサ数）。
            u_t (np.ndarray): 入力信号配列。
            t_seq (np.ndarray): 時刻配列。
            reset_state (bool, optional): 状態をリセットするかどうか。デフォルトはTrue。
            tqdm_title (str | None, optional): 進捗バーのタイトル。デフォルトはNone。

        Returns:
            tuple[np.ndarray, QRCStateArray]: 出力時刻配列とQRC状態系列。
        """
        return self._time_evol_series_computer.execute(
            solver=self._solver,
            n_mpx=n_mpx,
            u_t=u_t,
            t_arr=t_seq,
            reset_state=reset_state,
            tqdm_title=tqdm_title,
        )

    # TODO: ↕の_get_time_evol_statesのみをサービスとして切り出してテストを作り、一致を確認 非stiffなデータを使用しているから？

    def _check_state_count(self, states: AbstractState2DArray):
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

    def _compute_single_states(
            self,
            dataset: Dataset,
            *,
            tqdm_title: str | None = None,
    ) -> StateComputationResult:
        """
        Compute QRC state series for a single dataset.

        Args:
            dataset (Dataset): Input dataset containing time array, input signal, and target output.
            tqdm_title (str | None, optional): Title for progress bar. If None, no progress bar is shown.

        Returns:
            StateComputationResult: Result object containing computed state series and related data
                including time array, input array, x_seq_n, and target output array.
        """
        u_seq, y_seq, t_seq, u_t = dataset.u_seq, dataset.y_true_seq, dataset.t_seq, dataset.u_t
        valid_time_mask, x_seq_n = self._get_time_evol_states(
            n_mpx=self._param.n_mpx,
            u_t=u_t,
            t_seq=t_seq,
            reset_state=True,
            tqdm_title=tqdm_title,
        )
        self._check_state_count(x_seq_n)
        t_seq = t_seq[valid_time_mask]
        u_seq = u_seq[valid_time_mask]
        y_seq = y_seq[valid_time_mask]
        return StateComputationResult(
            t_seq=t_seq,
            u_seq_n=u_seq[:, None],
            x_seq_n=x_seq_n,
            y_seq_n=y_seq[:, None],
        )

    def _compute_states(
            self,
            datasets: Sequence[Dataset],
            *,
            tqdm_title: str | None = None,
    ) -> list[StateComputationResult]:
        """
        Compute QRC state series for multiple datasets in parallel.

        Args:
            datasets (Sequence[Dataset]): Input datasets containing time array, input signal, and target output.
            tqdm_title (str | None, optional): Title for progress bar. If None, no progress bar is shown.

        Returns:
            list[StateComputationResult]: List of result objects containing computed state series and related data.
        """
        if self._n_cpu == 1:
            return [
                self._compute_single_states(dataset, tqdm_title=tqdm_title)
                for dataset in datasets
            ]
        else:
            return Parallel(n_jobs=-1)(
                delayed(self._compute_single_states)(dataset, tqdm_title=tqdm_title)
                for dataset in datasets
            )

    def fit(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=True,
    ) -> list[PredictionResult]:
        """
        複数のデータセットを用いてモデルを学習する。

        Args:
            datasets (Sequence[Dataset]): 学習に用いるデータセットのシーケンス。
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはFalse。

        Returns:
            Self: 学習済みのインスタンス自身を返す。
        """
        # 複数のデータセットの状態を並列計算
        state_comp_results = self._compute_states(
            datasets,
            tqdm_title="fit" if show_progress else None
        )

        x_lst, y_lst = [], []
        for result in state_comp_results:
            x_seq_n = result.x_seq_n
            y_seq_n = result.y_seq_n
            x_lst.append(x_seq_n)
            y_lst.append(y_seq_n)

        # 全データセットを連結
        x_all = np.concatenate(x_lst, axis=0)
        y_all = np.concatenate(y_lst, axis=0)

        # 線形回帰モデルを学習
        self._lr_model.fit(x_all, y_all)

        # 予測
        results = self.predict(datasets, state_comp_result=state_comp_results)
        return results

    def predict(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=False,
            state_comp_result: Sequence[StateComputationResult | None] | None = None,
    ) -> list[PredictionResult]:
        """
        指定したデータセットに対して予測を行う。

        Args:
            datasets (Sequence[Dataset]): 予測対象のデータセットのシーケンス。
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはTrue。
            state_comp_result (list[StateComputationResult | None] | None, optional):
                事前に計算したStateComputationResultのリスト。
                Noneの場合は全て新たに計算される。
                リストの場合、Noneの要素に対応するデータセットのみ新たに計算される。
                デフォルトはNone。

        Returns:
            list[PredictionResult]: 予測結果を格納したPredictionResultインスタンスのリスト。
        """
        # QRC状態系列と出力時刻配列を取得
        if state_comp_result is None:
            # 全てのデータセットで計算
            state_comp_results = self._compute_states(
                datasets,
                tqdm_title="predict" if show_progress else None
            )
        else:
            # 一部のデータセットのみ計算が必要
            state_comp_results = list(state_comp_result)  # コピーを作成

            # Noneの要素を特定し、対応するデータセットを抽出
            datasets_to_compute = []
            indices_to_compute = []
            for i, result in enumerate(state_comp_results):
                if result is None:
                    datasets_to_compute.append(datasets[i])
                    indices_to_compute.append(i)

            # 必要なデータセットで計算を実行
            if datasets_to_compute:
                computed_results = self._compute_states(
                    datasets_to_compute,
                    tqdm_title="predict" if show_progress else None
                )

                # 計算結果を適切な位置に配置
                for idx, computed_result in zip(indices_to_compute, computed_results):
                    state_comp_results[idx] = computed_result

        # 各データセットに対して予測を実行
        results = []
        for result in state_comp_results:
            x_seq_n = result.x_seq_n
            y_true_seq_n = result.y_seq_n

            # 線形回帰モデルによる予測
            y_pred_seq_n = self._lr_model.predict(x_seq_n)

            # 予測結果をPredictionResultとして返す
            prediction_result = PredictionResult(
                t_seq=result.t_seq,
                u_seq_n=result.u_seq_n,
                x_seq_n=result.x_seq_n,
                y_true_seq_n=y_true_seq_n,
                y_pred_seq_n=y_pred_seq_n,
                n_washout=self._n_washout,
            )
            results.append(prediction_result)
        return results
