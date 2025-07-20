from dataclasses import dataclass
from functools import reduce

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from model import AbstractStateSeries, QRCParam, QRCStateTimeStep, QRCStateSeries, \
    QRCExperimentResultEntry, QRCExperimentResult
from physical_system import NVReservoirPhysicsSystem, NVReservoirObservable, \
    NVReservoirCollapseOperator
from time_evol_solver import TimeEvolutionSolver
from utils.axis import Axis
from utils.dataset import to_continuous_function_with_linspace_time, DelayedSine, LaggedInput
from utils.fullstate import fullstate


@dataclass(slots=True)
class ForwardResult:
    t: np.ndarray  # (T,)
    u: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y: np.ndarray  # (T, n_out)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t, np.ndarray), (type(self.t), self.t)
        assert isinstance(self.u, np.ndarray), (type(self.u), self.u)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y, np.ndarray), (type(self.y), self.y)

        n_t = len(self.t)
        assert self.t.ndim == 1, self.t.shape
        assert self.u.ndim == 2, self.u.shape
        assert self.u.shape[0] == n_t, (self.u.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y.ndim == 2, self.y.shape
        assert self.y.shape[0] == n_t, (self.y.shape[0], n_t)


@dataclass(slots=True)
class RegressionResult:
    t: np.ndarray  # (T,)
    u: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y: np.ndarray  # (T, n_out)
    y_pred: np.ndarray  # (T, n_out)
    r2_score: float

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t, np.ndarray), (type(self.t), self.t)
        assert isinstance(self.u, np.ndarray), (type(self.u), self.u)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y, np.ndarray), (type(self.y), self.y)
        assert isinstance(self.y_pred, np.ndarray), (type(self.y_pred), self.y_pred)

        n_t = len(self.t)
        assert self.t.ndim == 1, self.t.shape
        assert self.u.ndim == 2, self.u.shape
        assert self.u.shape[0] == n_t, (self.u.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y.ndim == 2, self.y.shape
        assert self.y.shape[0] == n_t, (self.y.shape[0], n_t)
        assert self.y_pred.ndim == 2, self.y_pred.shape
        assert self.y_pred.shape[0] == n_t, (self.y_pred.shape[0], n_t)
        assert self.r2_score is not None, self.r2_score


class QRCExperiment:
    """QRC実験を実行するクラス"""

    def __init__(
            self,
            *,
            system: NVReservoirPhysicsSystem,
            solver: TimeEvolutionSolver,
            dataset_train: list,
            dataset_test: list,
            n_samples_train: int,
            n_samples_test: int,
            n_mpx: int,
            n_washout: int,
    ):
        self._system = system
        self._solver = solver
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test
        self._lr_model = LinearRegression()

        self._n_samples_train = n_samples_train
        self._n_samples_test = n_samples_test
        self._n_mpx = n_mpx
        self._n_washout = n_washout

    @staticmethod
    def _create_oversampled_time_series_for_multiplex(dataset_gen, *, t_max, n_steps, n_mpx):
        """オーバーサンプルされた時系列データを作成"""
        t_arr = np.linspace(0, t_max, n_steps + 1)  # オーバーサンプルなし
        t_mpx_arr = np.linspace(0, t_max, n_steps * n_mpx + 1)  # オーバーサンプルあり
        assert np.allclose(t_arr, t_mpx_arr[::n_mpx])
        t_mpx_arr[::n_mpx] = t_arr  # avoid failure on np.isin
        u_mpx_arr, y_mpx_arr = dataset_gen(t_mpx_arr)
        u_t = to_continuous_function_with_linspace_time(t_mpx_arr, u_mpx_arr)
        mask = np.isin(t_mpx_arr, t_arr)
        assert np.count_nonzero(mask) == len(t_arr), (np.count_nonzero(mask), len(t_arr))
        u_arr, y_arr = u_mpx_arr[mask], y_mpx_arr[mask]
        assert len(u_arr) == len(y_arr) == len(t_arr), (len(u_arr), len(y_arr), len(t_arr))
        return u_arr, y_arr, t_arr, u_t

    @staticmethod
    def _create_constant_step_time_series(dataset_gen, *, t_max, n_steps):
        t_arr = np.linspace(0, t_max, n_steps + 1)
        u_arr, y_arr = dataset_gen(t_arr)
        u_t = to_continuous_function_with_linspace_time(t_arr, u_arr)
        return u_arr, y_arr, t_arr, u_t

    @classmethod
    def create_instance(cls, param: QRCParam):
        """パラメータからQRCExperimentインスタンスを作成"""
        # 物理系・ソルバーを構築
        system = NVReservoirPhysicsSystem(
            n_qubit=param.n_qubits,
            j_mean=param.j_mean,
            j_std=param.j_std,
            j_axis=Axis.X,
            h_mean=param.h_mean,
            h_std=param.h_std,
            h_axis=Axis.Z,
            h_td_axis=Axis.X,
            seed=param.seed,
        )

        solver = TimeEvolutionSolver(
            system=system,
            observable=reduce(lambda x, y: x + y, [
                NVReservoirObservable(n_qubit=param.n_qubits, axis=axis)
                for is_enabled, axis in
                [(param.obs_x, Axis.X), (param.obs_y, Axis.Y), (param.obs_z, Axis.Z)]
                if is_enabled
            ]),
            collapse_operator=NVReservoirCollapseOperator(n_qubit=param.n_qubits,
                                                          gamma_z=param.gamma_z),
            init_psi=fullstate(",".join(["z+"] * param.n_qubits)),
        )

        # データセットを生成
        rng = np.random.RandomState(seed=param.seed)
        if param.func_type == "lagged_sine":
            dataset = [
                cls._create_oversampled_time_series_for_multiplex(
                    lambda t_arr: DelayedSine.create(t_arr, rng=rng, lag=5 * param.n_mpx, freq=1.0),
                    t_max=param.t_max,
                    n_steps=param.n_steps,
                    n_mpx=param.n_mpx,
                ) for _ in range(param.n_samples_train + param.n_samples_test)
            ]
        elif param.func_type == "lagged_random_uniform":
            dataset = [
                cls._create_constant_step_time_series(
                    lambda t_arr: LaggedInput.create_with_random_input(
                        n=len(t_arr), rng=rng, lag=2,
                    ),
                    t_max=param.t_max,
                    n_steps=param.n_steps,
                ) for _ in range(param.n_samples_train + param.n_samples_test)
            ]
        else:
            raise ValueError(f"Invalid func_type: {param.func_type}")
        dataset_train = dataset[:param.n_samples_train]
        dataset_test = dataset[param.n_samples_train:]

        return cls(
            system=system,
            solver=solver,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            n_samples_train=param.n_samples_train,
            n_samples_test=param.n_samples_test,
            n_mpx=param.n_mpx,
            n_washout=param.n_washout,
        )

    def _get_time_evol_states(self, u_t, t_arr, *, tqdm_title: str | None):
        """時間発展状態を取得"""
        steps = []
        t_arr_out = []
        it = range(len(t_arr) - 2)
        if tqdm_title:
            it = tqdm(it, desc=tqdm_title)
        for i in it:  # mesolveが後方の時刻を参照するため少し前で止める
            # ステップの開始時刻から終了時刻まで時間発展させて、各時刻における結果を得る
            t_begin, t_end = t_arr[i], t_arr[i + 1]
            t_div = np.linspace(t_begin, t_end, self._n_mpx + 1)[:-1]  # shape: (n_mpx,)
            result = self._solver.forward(u_t, t_div)

            # flatten all qubit states at this time step
            states = np.stack([result.expect(j) for j in range(result.n_expect)],
                              axis=1)  # (n_mpx, 状態数)
            steps.append(QRCStateTimeStep(states=states))
            t_arr_out.append(t_begin)

        return np.array(t_arr_out)[self._n_washout:], QRCStateSeries(steps=steps)[self._n_washout:]

    def _forward_data(self, dataset: list, *, show_progress=False, name_prefix: str):
        """データを処理"""
        forward_results: list[ForwardResult] = []
        for i in range(len(dataset)):
            u_arr, y_arr, t_arr, u_t = dataset[i]

            # QRC状態系列を計算
            self._solver.reset_rho()
            t_arr_out, states = self._get_time_evol_states(
                u_t, t_arr, tqdm_title=f"{name_prefix} #{i}" if show_progress else None,
            )
            mask = np.isin(t_arr, t_arr_out)
            t_arr, u_arr, y_arr = t_arr[mask], u_arr[mask], y_arr[mask]

            forward_results.append(
                ForwardResult(
                    t=t_arr,
                    u=u_arr[:, None],
                    states=states,
                    y=y_arr[:, None],
                )
            )
        return forward_results

    def _train_regression_model(self, train_forward_results):
        """回帰モデルを訓練"""
        state_all, y_all = [], []
        for i in range(self._n_samples_train):
            state_all.append(np.array(train_forward_results[i].states))
            y_all.append(train_forward_results[i].y)
        state_all = np.concatenate(state_all, axis=0)  # (T * n_samples, n_states)
        y_all = np.concatenate(y_all, axis=0)  # (T * n_samples, n_out)

        self._lr_model.fit(state_all, y_all)

    def _evaluate_data(self, forward_results: list[ForwardResult]):
        """データを評価"""
        regression_results: list[RegressionResult] = []
        for i in range(len(forward_results)):
            forward_result = forward_results[i]
            y_pred = self._lr_model.predict(np.array(forward_result.states))
            r2_score = self._lr_model.score(np.array(forward_result.states), forward_result.y)
            regression_results.append(
                RegressionResult(
                    t=forward_result.t,
                    u=forward_result.u,
                    states=forward_result.states,
                    y=forward_result.y,
                    y_pred=y_pred,
                    r2_score=r2_score,
                )
            )
        return regression_results

    # noinspection DuplicatedCode
    def run(self, *, show_progress=False) \
            -> tuple[QRCExperimentResultEntry, QRCExperimentResultEntry]:
        """QRC実験を実行"""
        # 訓練データ処理
        train_forward_results = self._forward_data(
            self._dataset_train,
            show_progress=show_progress,
            name_prefix="train",
        )

        # 回帰モデル訓練
        self._train_regression_model(train_forward_results)

        # 訓練データ評価
        train_regression_results = self._evaluate_data(train_forward_results)

        # テストデータ処理
        test_forward_results = self._forward_data(
            self._dataset_test,
            show_progress=show_progress,
            name_prefix="test",
        )

        # テストデータ評価
        test_regression_results = self._evaluate_data(test_forward_results)

        # 結果を返す
        train_entry = QRCExperimentResultEntry(
            name=f"train",
            t=np.array([r.t for r in train_regression_results]),
            u=np.array([r.u for r in train_regression_results]),
            states=[r.states for r in train_regression_results],
            y_pred=np.array([r.y_pred for r in train_regression_results]),
            y_true=np.array([r.y for r in train_regression_results]),
            r2_score=np.array([r.r2_score for r in train_regression_results]),
        )

        test_entry = QRCExperimentResultEntry(
            name=f"test",
            t=np.array([r.t for r in test_regression_results]),
            u=np.array([r.u for r in test_regression_results]),
            states=[r.states for r in test_regression_results],
            y_pred=np.array([r.y_pred for r in test_regression_results]),
            y_true=np.array([r.y for r in test_regression_results]),
            r2_score=np.array([r.r2_score for r in test_regression_results]),
        )

        return train_entry, test_entry


def run_qrc_experiment(p: QRCParam, *, show_progress=False) -> QRCExperimentResult:
    """QRC実験を実行する関数（後方互換性のため）"""
    experiment = QRCExperiment.create_instance(p)
    train_entry, test_entry = experiment.run(show_progress=show_progress)
    return QRCExperimentResult(
        param=p,
        train=train_entry,
        test=test_entry,
    )
