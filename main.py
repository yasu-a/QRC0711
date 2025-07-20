from dataclasses import dataclass, asdict
from datetime import datetime
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from model import AbstractStateSeries, QRCStateTimeStep, QRCStateSeries
from model import QRCExperimentResult, QRCExperimentResultEntry, QRCParam
from physical_system import NVReservoirObservable, NVReservoirCollapseOperator, \
    NVReservoirPhysicsSystem
from time_evol_solver import TimeEvolutionSolver
from utils.axis import Axis
from utils.dataset import DelayedSine, to_continuous_function_with_linspace_time
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

        from functools import reduce
        solver = TimeEvolutionSolver(
            system=system,
            observable=reduce(lambda x, y: x + y, [
                NVReservoirObservable(n_qubit=param.n_qubits, axis=axis)
                for is_enabled, axis in [(param.obs_x, Axis.X), (param.obs_y, Axis.Y), (param.obs_z, Axis.Z)]
                if is_enabled
            ]),
            collapse_operator=NVReservoirCollapseOperator(n_qubit=param.n_qubits,
                                                          gamma_z=param.gamma_z),
            init_psi=fullstate(",".join(["z+"] * param.n_qubits)),
        )

        # データセットを生成
        rng = np.random.RandomState(seed=param.seed)
        dataset = [
            cls._create_oversampled_time_series_for_multiplex(
                lambda t_arr: DelayedSine.create(t_arr, rng=rng, lag=10 * param.n_mpx,
                                                 freq=1.0),
                t_max=param.t_max,
                n_steps=param.n_steps,
                n_mpx=param.n_mpx,
            ) for _ in range(param.n_samples_train + param.n_samples_test)
        ]
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


# パラメータ探索グリッド
def param_mapper_fn(d: dict) -> QRCParam:
    return QRCParam(
        n_qubits=int(d["n_qubits"]),
        gamma_z=float(d["gamma_z"]),
        n_mpx=int(d["n_mpx"]),
        j_mean=float(d["j_mean"]),
        j_std=float(d["j_mean"] / 2),
        h_mean=float(d["h_mean"]),
        h_std=float(d["h_mean"] / 2),
        obs_x=bool(d["obs_x"]),
        obs_y=bool(d["obs_y"]),
        obs_z=bool(d["obs_z"]),
        n_steps=int(d["n_steps"]),
        t_max=float(d["t_max"]),
        test_ratio=float(d["test_ratio"]),
        n_washout=int(d["n_washout"]),
        n_samples_train=int(d["n_samples_train"]),
        n_samples_test=int(d["n_samples_test"]),
        seed=int(d["seed"]),
    )


def scorer_fn(p: QRCParam) -> float:
    return run_qrc_experiment(p).test.r2_score_avg


def main():
    from search.ga import GAParameterSearcher
    searcher = GAParameterSearcher[QRCParam](
        scorer=scorer_fn,
        param_grid=dict(
            n_qubits=[4],
            gamma_z=[0.1, 0.01, 0.001, 0.0001, 0.00001],
            n_mpx=[1, 2, 4, 6, 8, 10],
            j_mean=np.arange(0.1, 2.0, 0.1),
            h_mean=np.arange(0.1, 2.0, 0.1),
            obs_x=[False, True],
            obs_y=[False, True],
            obs_z=[False, True],
            n_steps=[150],
            t_max=[5],
            test_ratio=[0.5],
            n_washout=[5, 10, 20, 50],
            n_samples_train=[8],
            n_samples_test=[4],
            seed=[0],
        ),
        param_mapper=param_mapper_fn,
        forbid=[
            dict(
                obs_x=False,
                obs_y=False,
                obs_z=False,
            )
        ]
    )
    best_param, best_score = searcher.search(n_workers=9)
    
    print(f"Best param: {best_param} with R^2={best_score}")
    import pandas as pd
    df = pd.DataFrame([{**asdict(p), "_score": score} for p, score in searcher.history])
    df = df.sort_values('_score', ascending=False)
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df)
    os.makedirs('./results', exist_ok=True)
    df.to_csv(f'./results/qrc_param_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

    # 最良パラメータで実験・グラフ表示
    # best_param = QRCParam(
    #     n_qubits=4,
    #     gamma_z=0.001,
    #     n_mpx=4,
    #     j_mean=1.0,
    #     j_std=0.3,
    #     h_mean=1.0,
    #     h_std=0.3,
    #     n_steps=150,
    #     obs_x=True,
    #     obs_y=True,
    #     obs_z=True,
    #     t_max=5,
    #     test_ratio=0.5,
    #     n_washout=20,
    #     n_samples_train=8,
    #     n_samples_test=4,
    #     seed=0,
    # )
    best_param = QRCParam(n_qubits=4, gamma_z=0.0001, n_mpx=4, j_mean=0.6, j_std=0.3, h_mean=0.9,
                          h_std=0.45, n_steps=150, obs_x=False, obs_y=True, obs_z=True, t_max=5.0,
                          test_ratio=0.5, n_washout=5, n_samples_train=8, n_samples_test=4, seed=0)
    results = run_qrc_experiment(best_param, show_progress=True)
    # results.plot_state_series()
    results.plot_prediction()


if __name__ == '__main__':
    main()

r"""
C:\Users\yasuh\PycharmProjects\QRC0711\.venv\Scripts\python.exe C:\Users\yasuh\PycharmProjects\QRC0711\main.py 
Best R2=0.908, param={'N_QUBITS': 6, 'GAMMA_Z': 0.001, 'V': 8, 'J_MEAN': 0.5, 'J_STD': 1.0, 'H_MEAN': 2.5, 'H_STD': 1.75, 'M': 500, 'T': 30, 'test_ratio': 0.3, 'washout': 10, 'random_seed': 0}:  28%|██▊       | 274/972 [8:26:29<783:50:16, 4042.72s/it]

0.5     2.5
/6C2    /6x2
0.0333  0.2083
"""
