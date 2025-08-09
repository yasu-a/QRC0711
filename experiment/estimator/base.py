from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from core.time_evol_solver import AbstractTimeEvolutionSolver
from experiment.estimator.dto import StateComputationResult
from model.dataset import Continuous, Discrete, Dataset
from model.physical_system import AbstractResponsivePhysicalSystem
from model.prediction_result import PredictionResult
from model.state_array import QRCStateArray, AbstractState2DArray
from service.compute_time_evol import get_compute_time_evol_state_series_service


class AbstractEstimator(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def fit(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=False,
    ) -> list[PredictionResult]:
        raise NotImplementedError()

    @abstractmethod
    def predict(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=False,
            state_comp_result: Sequence[StateComputationResult | None] | None = None,
    ) -> list[PredictionResult]:
        raise NotImplementedError()


class AbstractReservoirEstimator(AbstractEstimator):
    """
    量子リザバーコンピューティング推定器の抽象基底クラス。
    
    NVQRC、FNQRC等の共通機能を提供する。
    """

    def __init__(
            self,
            *,
            system: AbstractResponsivePhysicalSystem,
            solver: AbstractTimeEvolutionSolver,
            seed: int,
            n_washout: int,
            n_cpu: int,
    ) -> None:
        """
        AbstractReservoirEstimatorを初期化する。

        Args:
            system (AbstractResponsivePhysicalSystem): 物理システム
            solver (AbstractTimeEvolutionSolver): time evolution solver
            seed (int): 乱数シード
            n_washout (int): ウォッシュアウト期間
            n_cpu (int): 並列処理用CPU数
        """
        assert n_cpu == -1 or n_cpu >= 1, f"invalid {n_cpu=}"

        self._system = system
        self._solver = solver
        self._seed = seed
        self._n_washout = n_washout
        self._n_cpu = n_cpu

        self._lr_model = LinearRegression()
        self._time_evol_series_computer = get_compute_time_evol_state_series_service()

    @abstractmethod
    def _get_time_evol_states(
            self,
            *,
            u_t: Continuous,
            t_seq: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None
    ) -> tuple[np.ndarray, QRCStateArray]:
        """
        QRC状態系列と出力時刻配列を計算する。サブクラスで実装する。

        Args:
            u_t (np.ndarray): 入力信号配列
            t_seq (np.ndarray): 時刻配列
            reset_state (bool, optional): 状態をリセットするかどうか。デフォルトはTrue。
            tqdm_title (str | None, optional): 進捗バーのタイトル。デフォルトはNone。

        Returns:
            tuple[np.ndarray, QRCStateArray]: 出力時刻配列とQRC状態系列。
        """
        raise NotImplementedError()

    @abstractmethod
    def _check_state_count(self, states: AbstractState2DArray) -> None:
        """状態数の妥当性をチェックする。サブクラスで実装する。"""
        raise NotImplementedError()

    def _compute_single_states(
            self,
            dataset: Dataset,
            *,
            tqdm_title: str | None = None,
    ) -> StateComputationResult:
        """
        単一データセットのQRC状態系列を計算する。

        Args:
            dataset (Dataset): 入力データセット（時刻配列、入力信号、目標出力を含む）
            tqdm_title (str | None, optional): 進捗バーのタイトル。Noneの場合は進捗バーを表示しない。

        Returns:
            StateComputationResult: 計算された状態系列と関連データを含む結果オブジェクト
        """
        u_seq, y_seq, t_seq, u_t = dataset.u_seq, dataset.y_true_seq, dataset.t_seq, dataset.u_t
        valid_time_mask, x_seq_n = self._get_time_evol_states(
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
        複数データセットのQRC状態系列を並列で計算する。

        Args:
            datasets (Sequence[Dataset]): 入力データセット群
            tqdm_title (str | None, optional): 進捗バーのタイトル。Noneの場合は進捗バーを表示しない。

        Returns:
            list[StateComputationResult]: 計算された状態系列と関連データを含む結果オブジェクトのリスト。
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
            datasets (Sequence[Dataset]): 学習に用いるデータセットのシーケンス
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはTrue。

        Returns:
            list[PredictionResult]: 学習用データに対する予測結果のリスト
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
            datasets (Sequence[Dataset]): 予測対象のデータセットのシーケンス
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはFalse。
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
