from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce, cache
from typing import Callable, Iterable

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from model import AbstractStateSeries, NVQRCParam, QRCStateTimeStep, QRCStateSeries
from physical_system import NVReservoirPhysicsSystem, NVReservoirObservable, \
    NVReservoirCollapseOperator, AbstractPhysicalSystem
from time_evol_solver import TimeEvolutionSolver
from utils.axis import Axis
from utils.dataset_v2 import AbstractDatasetGenerator, Dataset, Discrete, Continuous
from utils.fullstate import fullstate
from utils.score import score_func_by_name, ScoreName


@dataclass(slots=True)
class StateComputationResult:
    t_arr: np.ndarray  # (T,)
    u_mlt_arr: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y_mlt_arr: np.ndarray  # (T, n_out)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t_arr, np.ndarray), (type(self.t_arr), self.t_arr)
        assert isinstance(self.u_mlt_arr, np.ndarray), (type(self.u_mlt_arr), self.u_mlt_arr)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y_mlt_arr, np.ndarray), (type(self.y_mlt_arr), self.y_mlt_arr)

        n_t = len(self.t_arr)
        assert self.t_arr.ndim == 1, self.t_arr.shape
        assert self.u_mlt_arr.ndim == 2, self.u_mlt_arr.shape
        assert self.u_mlt_arr.shape[0] == n_t, (self.u_mlt_arr.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y_mlt_arr.ndim == 2, self.y_mlt_arr.shape
        assert self.y_mlt_arr.shape[0] == n_t, (self.y_mlt_arr.shape[0], n_t)

        # copy arrays and make readonly
        self.t_arr = self.t_arr.copy()
        self.t_arr.setflags(write=False)
        self.u_mlt_arr = self.u_mlt_arr.copy()
        self.u_mlt_arr.setflags(write=False)
        self.y_mlt_arr = self.y_mlt_arr.copy()
        self.y_mlt_arr.setflags(write=False)


@dataclass(slots=True)
class PredictionResult:
    t_arr: np.ndarray  # (T,)
    u_mlt_arr: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y_mlt_arr: np.ndarray  # (T, n_out)
    y_pred_mlt_arr: np.ndarray  # (T, n_out)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t_arr, np.ndarray), (type(self.t_arr), self.t_arr)
        assert isinstance(self.u_mlt_arr, np.ndarray), (type(self.u_mlt_arr), self.u_mlt_arr)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y_mlt_arr, np.ndarray), (type(self.y_mlt_arr), self.y_mlt_arr)
        assert isinstance(self.y_pred_mlt_arr, np.ndarray), (type(self.y_pred_mlt_arr),
                                                             self.y_pred_mlt_arr)

        n_t = len(self.t_arr)
        assert self.t_arr.ndim == 1, self.t_arr.shape
        assert self.u_mlt_arr.ndim == 2, self.u_mlt_arr.shape
        assert self.u_mlt_arr.shape[0] == n_t, (self.u_mlt_arr.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y_mlt_arr.ndim == 2, self.y_mlt_arr.shape
        assert self.y_mlt_arr.shape[0] == n_t, (self.y_mlt_arr.shape[0], n_t)
        assert self.y_pred_mlt_arr.ndim == 2, self.y_pred_mlt_arr.shape
        assert self.y_pred_mlt_arr.shape[0] == n_t, (self.y_pred_mlt_arr.shape[0], n_t)

        # copy arrays and make readonly
        self.t_arr = self.t_arr.copy()
        self.t_arr.setflags(write=False)
        self.u_mlt_arr = self.u_mlt_arr.copy()
        self.u_mlt_arr.setflags(write=False)
        self.y_mlt_arr = self.y_mlt_arr.copy()
        self.y_mlt_arr.setflags(write=False)
        self.y_pred_mlt_arr = self.y_pred_mlt_arr.copy()
        self.y_pred_mlt_arr.setflags(write=False)

    @cache
    def score(
            self,
            score: ScoreName,
    ) -> np.ndarray:
        """
        このPredictionResultインスタンスの予測結果に対するスコアを計算する。

        Args:
            score (Literal["r2", "capacity", "mse"]): 計算するスコアの種類

        Returns:
            np.ndarray: スコア値（shape: (n_out,)）
        """
        score_func = score_func_by_name(score)
        return score_func(y_true=self.y_mlt_arr, y_pred=self.y_pred_mlt_arr)


class PredictionResultSet:
    def __init__(self, results: list[PredictionResult]):
        self._results = results

    def __len__(self):
        return len(self._results)

    def __getitem__(self, sample_index: int):
        return self._results[sample_index]

    @cache
    def aggregated_score(
            self,
            score: ScoreName,
    ) -> np.ndarray:
        """
        結果全体のデータをひとつのデータとして集約し、scoreを計算する。
        """
        y_true_arr = np.array([r.y_mlt_arr for r in self._results])
        y_pred_arr = np.array([r.y_pred_mlt_arr for r in self._results])
        score_func = score_func_by_name(score)
        return score_func(y_true=y_true_arr, y_pred=y_pred_arr)


class AbstractEstimator(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, datasets: Iterable[Dataset], *, show_progress=False) -> list[PredictionResult]:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset: Dataset, *, show_progress=False,
                state_comp_result: StateComputationResult = None) -> PredictionResult:
        raise NotImplementedError()


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
    def _create_solver(cls, *, param: NVQRCParam, system: AbstractPhysicalSystem):
        return TimeEvolutionSolver(
            system=system,
            observable=reduce(
                lambda x, y: x + y,
                [
                    NVReservoirObservable(n_qubit=param.n_qubits, axis=axis)
                    for is_enabled, axis in
                    [(param.obs_x, Axis.X), (param.obs_y, Axis.Y), (param.obs_z, Axis.Z)]
                    if is_enabled
                ],
            ),
            collapse_operator=NVReservoirCollapseOperator(n_qubit=param.n_qubits,
                                                          gamma_z=param.gamma_z),
            init_psi=fullstate(",".join(["z+"] * param.n_qubits)),
        )

    def __init__(self, *, param: NVQRCParam, seed: int, n_washout: int):
        self._param = param
        self._seed = seed
        self._n_washout = n_washout
        self._lr_model = LinearRegression()
        self._system = self._create_system(param=param, seed=seed)
        self._solver = self._create_solver(param=param, system=self._system)

    @classmethod
    def _get_time_evol_states(
            cls,
            *,
            solver: TimeEvolutionSolver,
            n_mpx: int,
            u_t: Continuous,
            t_arr: Discrete,
            reset_state: bool = True,
            tqdm_title: str | None = None
    ) -> tuple[np.ndarray, QRCStateSeries]:
        """
        QRC状態系列と出力時刻配列を計算する。

        Args:
            solver (TimeEvolutionSolver): 時間発展を計算するソルバー。
            n_mpx (int): 各時刻区間を分割する数（マルチプレクサ数）。
            u_t (np.ndarray): 入力信号配列。
            t_arr (np.ndarray): 時刻配列。
            reset_state (bool, optional): 状態をリセットするかどうか。デフォルトはTrue。
            tqdm_title (str | None, optional): 進捗バーのタイトル。デフォルトはNone。

        Returns:
            tuple[np.ndarray, QRCStateSeries]: 出力時刻配列とQRC状態系列。
        """
        # QRC状態と出力時刻配列の初期化
        steps: list[QRCStateTimeStep] = []

        # 状態をリセットする場合
        if reset_state:
            solver.reset_rho()

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
            t_div = np.linspace(t_begin, t_end, n_mpx + 1)[:-1]
            result = solver.forward(u_t, t_div)

            # 各観測量の期待値をまとめて配列化（shape: (n_mpx, n_expect)）
            states = np.stack([result.expect(j) for j in range(result.n_expect)], axis=1)
            steps.append(QRCStateTimeStep(states=states))

        # 出力時刻配列とQRC状態列を返す
        return valid_time_mask, QRCStateSeries(steps=steps)

    def _compute_states(self, dataset: Dataset, *,
                        tqdm_title: str | None = None) -> StateComputationResult:
        """
        Compute QRC state series for the specified dataset.

        Args:
            dataset (Dataset): Input dataset containing time array, input signal, and target output.
            tqdm_title (str | None, optional): Title for progress bar. If None, no progress bar is shown.

        Returns:
            StateComputationResult: Result object containing computed state series and related data
                including time array, input array, states, and target output array.
        """
        u_arr, y_arr, t_arr, u_t = dataset.u_arr, dataset.y_arr, dataset.t_arr, dataset.u_t
        valid_time_mask, states = self._get_time_evol_states(
            solver=self._solver,
            n_mpx=self._param.n_mpx,
            u_t=u_t,
            t_arr=t_arr,
            reset_state=True,
            tqdm_title=tqdm_title,
        )
        t_arr = t_arr[valid_time_mask]
        u_arr = u_arr[valid_time_mask]
        y_arr = y_arr[valid_time_mask]
        return StateComputationResult(
            t_arr=t_arr,
            u_mlt_arr=u_arr[:, None],
            states=states,
            y_mlt_arr=y_arr[:, None],
        )

    def fit(self, datasets: Iterable[Dataset], *, show_progress=True) -> list[PredictionResult]:
        """
        複数のデータセットを用いてモデルを学習する。

        Args:
            datasets (Iterable[Dataset]): 学習に用いるデータセットのイテラブル。
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはFalse。

        Returns:
            Self: 学習済みのインスタンス自身を返す。
        """
        x_lst, y_lst, state_comp_results = [], [], []
        for i, dataset in enumerate(datasets):
            result = self._compute_states(dataset, tqdm_title="fit" if show_progress else None)
            x_mlt_arr = np.array(result.states)
            y_mlt_arr = result.y_mlt_arr
            state_comp_results.append(result)
            x_lst.append(x_mlt_arr)
            y_lst.append(y_mlt_arr)

        # 全データセットを連結
        x_all = np.concatenate(x_lst, axis=0)
        y_all = np.concatenate(y_lst, axis=0)

        # 線形回帰モデルを学習
        self._lr_model.fit(x_all, y_all)

        # 予測
        results = []
        for dataset, state_comp_result in zip(datasets, state_comp_results):
            result = self.predict(dataset, state_comp_result=state_comp_result)
            results.append(result)
        return results

    def predict(self, dataset: Dataset, *, show_progress=True,
                state_comp_result: StateComputationResult = None) -> PredictionResult:
        """
        指定したデータセットに対して予測を行う。

        Args:
            dataset (Dataset): 予測対象のデータセット。
            show_progress (bool, optional): 進捗バーを表示するかどうか。デフォルトはFalse。

        Returns:
            PredictionResult: 予測結果を格納したRegressionResultインスタンス。
        """
        # QRC状態系列と出力時刻配列を取得し、出力時刻に対応するデータのみ抽出
        if state_comp_result is None:
            result = self._compute_states(dataset, tqdm_title="predict" if show_progress else None)
        else:
            result = state_comp_result
        x_mlt_arr = np.array(result.states)
        y_mlt_arr = result.y_mlt_arr

        # 線形回帰モデルによる予測
        y_pred_arr = self._lr_model.predict(x_mlt_arr)

        # 予測結果をRegressionResultとして返す
        return PredictionResult(
            t_arr=result.t_arr,
            u_mlt_arr=result.u_mlt_arr,
            states=result.states,
            y_mlt_arr=y_mlt_arr,
            y_pred_mlt_arr=y_pred_arr,
        )


class AbstractExperimentSuite(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()


class PredictionExperimentSuite(AbstractExperimentSuite):
    """任意の予測実験Modelをまとめて管理・評価する汎用Suiteクラス"""

    def __init__(
            self,
            *,
            generator_fn: Callable[[np.random.RandomState], AbstractDatasetGenerator],
            n_train_samples: int,
            n_test_samples: int,
            rng: np.random.RandomState,
            model_class: type[AbstractEstimator],
            model_kwargs: dict,
            show_progress: bool = False,
    ):
        self._generator_fn = generator_fn
        self._n_train_samples = n_train_samples
        self._n_test_samples = n_test_samples
        self._rng = rng
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._show_progress = show_progress

        self._run = False

        self._results_train: PredictionResultSet | None = None
        self._results_test: PredictionResultSet | None = None

    def run(self):
        if self._run:
            raise ValueError(
                "run() method has already been called. Cannot run experiment multiple times."
            )

        # Generate training datasets
        datasets_train = []
        for _ in range(self._n_train_samples):
            dataset = self._generator_fn(self._rng).create()
            datasets_train.append(dataset)

        # Generate test datasets  
        datasets_test = []
        for _ in range(self._n_test_samples):
            dataset = self._generator_fn(self._rng).create()
            datasets_test.append(dataset)

        # Create and fit model on training data
        model = self._model_class(**self._model_kwargs)
        results_train = model.fit(datasets_train, show_progress=self._show_progress)
        self._results_train = PredictionResultSet(results_train)

        # Get predictions on test data
        results_test = []
        for dataset in datasets_test:
            result = model.predict(dataset, show_progress=self._show_progress)
            results_test.append(result)
        self._results_test = PredictionResultSet(results_test)

        self._run = True

    def _check_run(self) -> None:
        if not self._run:
            raise RuntimeError("You must call run() before accessing results_train.")

    @property
    def results_train(self) -> PredictionResultSet:
        self._check_run()
        return self._results_train

    @property
    def results_test(self) -> PredictionResultSet:
        self._check_run()
        return self._results_test
