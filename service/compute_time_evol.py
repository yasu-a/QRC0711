from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from core.time_evol_solver import AbstractTimeEvolutionSolver
from model.dataset import Discrete, Continuous
from model.state_array import QRCStateTimeStep, QRCStateArray


class AbstractComputeTimeEvolStateSeriesService(ABC):
    @abstractmethod
    def execute(
            self,
            *,
            solver: AbstractTimeEvolutionSolver,
            n_mpx: int,
            u_t: Continuous,
            t_arr: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None,
    ) -> tuple[np.ndarray, QRCStateArray]:
        """
        Compute QRC state series and output time array.

        Args:
            solver (AbstractTimeEvolutionSolver): Time evolution solver
            n_mpx (int): Number of divisions for each time interval (multiplexer number)
            u_t (Continuous): Input signal function
            t_arr (Discrete): Time array
            reset_state (bool, optional): Whether to reset state. Defaults to True
            tqdm_title (str | None, optional): Title for progress bar. Defaults to None

        Returns:
            tuple[np.ndarray, QRCStateArray]: Valid time mask array and QRC state series
        """

        """   
        ====================================
         STATE SAMPLING EXAMPLE (n_mpx = 5)
        ====================================
        
           >--< τ
           |  |       
           |  #0 #1 #2 #3 #4
           +--+--+--+--+--+-- ... --> t
           |              |
           T
           
           X(T) = [ x'(#0), x'(#1), x'(#2), x'(#3), x'(#4) ]
                = [ x'(T + (1/5)τ), x'(T + (2/5)τ), x'(T + (3/5)τ), x'(T + (4/5)τ), x'(T + τ) ]
        """
        raise NotImplementedError()


class ComputeTimeEvolStateSeriesDividedForwardService(AbstractComputeTimeEvolStateSeriesService):
    def __init__(
            self,
    ):
        pass

    def execute(
            self,
            *,
            solver: AbstractTimeEvolutionSolver,
            n_mpx: int,
            u_t: Continuous,
            t_arr: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None,
    ) -> tuple[np.ndarray, QRCStateArray]:
        # QRC状態と出力時刻配列の初期化
        steps: list[QRCStateTimeStep] = []

        # 状態をリセットする場合
        if reset_state:
            solver.reset_state()

        valid_time_mask = np.zeros(len(t_arr), dtype=bool)
        valid_time_mask[:-2] = True  # mesolveが後方の時刻を参照するため少し前で止める

        # 進捗バーの設定（必要な場合）
        it = range(np.count_nonzero(valid_time_mask))
        if tqdm_title:
            it = tqdm(it, desc=tqdm_title)

        # 各時刻区間ごとに時間発展を計算し、QRC状態を記録
        for i in it:
            # ステップの開始時刻から終了時刻まで時間発展させて、各時刻における結果を得る
            t_begin, t_end = t_arr[i], t_arr[i + 1]
            t_div = np.linspace(t_begin, t_end, n_mpx + 1)
            result = solver.forward(u_t, t_div)
            # 各観測量の期待値をまとめて配列化（shape: (n_mpx, n_expect)）
            states = np.stack([result.expect(j)[1:] for j in range(result.n_expect)], axis=1)
            steps.append(QRCStateTimeStep(states=states))

        # 出力時刻配列とQRC状態列を返す
        return valid_time_mask, QRCStateArray(steps=steps)


class ComputeTimeEvolStateSeriesSingleForwardService(AbstractComputeTimeEvolStateSeriesService):
    def __init__(
            self,
    ):
        pass

    def execute(
            self,
            *,
            solver: AbstractTimeEvolutionSolver,
            n_mpx: int,
            u_t: Continuous,
            t_arr: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None,
    ) -> tuple[np.ndarray, QRCStateArray]:
        # NOTE
        # ====
        #
        #  t[t]～t[t+Δt]の間、入力u[t]からの相互作用を受けた情報を含むのは状態x[t+Δt]
        #  x[t+Δt]を使ってy[t]を当てる必要がある
        #   -> 論文ではu[t]で時間発展後の状態をx[t]と呼んでいる

        # 状態をリセットする場合
        if reset_state:
            solver.reset_state()

        # 有効な時刻配列のみを取得
        t_all = np.concatenate(
            [
                np.linspace(s, t, n_mpx + 1)[:-1]
                for s, t in zip(t_arr[:-1], t_arr[1:])
            ] + [
                [t_arr[-1]]
            ]
        )

        # 時間発展を解く
        result = solver.forward(u_t, t_all, pbar_title=tqdm_title)

        # 結果を各ステップごとに分割
        steps: list[QRCStateTimeStep] = []
        for i in range(n_mpx, len(t_all), n_mpx):
            # 各観測量の期待値をまとめて配列化（shape: (n_mpx, n_expect)）
            states = np.stack(
                [
                    result.expect(j)[i - n_mpx + 1:i + 1]
                    for j in range(result.n_expect)
                ],
                axis=1,
            )
            steps.append(QRCStateTimeStep(states=states))

        # 出力時刻配列とQRC状態列を返す
        valid_time_mask = np.zeros(len(t_arr), dtype=bool)
        valid_time_mask[:-1] = True
        return valid_time_mask, QRCStateArray(steps=steps)


def get_compute_time_evol_state_series_service() -> AbstractComputeTimeEvolStateSeriesService:
    # return ComputeTimeEvolStateSeriesDividedForwardService()
    return ComputeTimeEvolStateSeriesSingleForwardService()
