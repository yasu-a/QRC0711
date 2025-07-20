from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from model import QRCExperimentResult, QRCStateTimeStep, QRCStateSeries, QRCExperimentResultEntry, \
    QRCParam, AbstractStateSeries
from physical_system import NVReservoirObservable, NVReservoirCollapseOperator, \
    NVReservoirPhysicsSystem
from search.ga import GAParameterSearcher
from time_evol_solver import TimeEvolutionSolver
from utils.axis import Axis
from utils.dataset import DelayedSine, to_continuous_function_with_linspace_time
from utils.fullstate import fullstate


# QRC状態系列の計算関数
def get_time_evol_states(
        n_mpx, u_t, t_arr, solver, n_washout, *, tqdm_title: str | None
) -> tuple[np.ndarray, QRCStateSeries]:
    # ^ t_arrはu_arrやstatesに直接対応する（time-multiplexingを考慮しない）時刻の配列
    steps = []
    t_arr_out = []
    it = range(len(t_arr) - 2)
    if tqdm_title:
        it = tqdm(it, desc=tqdm_title)
    for i in it:  # mesolveが後方の時刻を参照するため少し前で止める
        # ステップの開始時刻から終了時刻まで時間発展させて、各時刻における結果を得る
        t_begin, t_end = t_arr[i], t_arr[i + 1]
        t_div = np.linspace(t_begin, t_end, n_mpx + 1)[:-1]  # shape: (n_mpx,)
        result = solver.forward(u_t, t_div)

        # flatten all qubit states at this time step
        states = np.stack([result.expect(j) for j in range(result.n_expect)],
                          axis=1)  # (n_mpx, 状態数)
        steps.append(QRCStateTimeStep(states=states))
        t_arr_out.append(t_begin)

    return np.array(t_arr_out)[n_washout:], QRCStateSeries(steps=steps)[n_washout:]


def create_oversampled_time_series_for_multiplex(dataset_gen, *, t_max, n_steps, n_mpx):
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


@dataclass(slots=True)
class ForwardResult:
    t: np.ndarray  # (T,)
    u: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y: np.ndarray  # (T, n_out)

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


def run_qrc_experiment(p: QRCParam, *, show_progress=False) -> QRCExperimentResult:
    # 物理系・ソルバーの構築
    system = NVReservoirPhysicsSystem(
        n_qubit=p.n_qubits,
        j_mean=p.j_mean,
        j_std=p.j_std,
        j_axis=Axis.X,
        h_mean=p.h_mean,
        h_std=p.h_std,
        h_axis=Axis.Z,
        h_td_axis=Axis.X,
        seed=p.seed,
    )
    solver = TimeEvolutionSolver(
        system=system,
        observable=(
                NVReservoirObservable(n_qubit=p.n_qubits, axis=Axis.X)
                + NVReservoirObservable(n_qubit=p.n_qubits, axis=Axis.Y)
                + NVReservoirObservable(n_qubit=p.n_qubits, axis=Axis.Z)
        ),
        collapse_operator=NVReservoirCollapseOperator(n_qubit=p.n_qubits, gamma_z=p.gamma_z),
        init_psi=fullstate(",".join(["x+"] * p.n_qubits)),
    )

    # GENERATE DATASET
    rng = np.random.RandomState(seed=p.seed)
    dataset = [
        create_oversampled_time_series_for_multiplex(
            lambda t_arr: DelayedSine.create(t_arr, rng=rng, lag=3 * p.n_mpx, freq=1.0),
            t_max=p.t_max,
            n_steps=p.n_steps,
            n_mpx=p.n_mpx,
        ) for _ in range(p.n_samples_train + p.n_samples_test)
    ]
    dataset_train = dataset[:p.n_samples_train]
    dataset_test = dataset[p.n_samples_train:]

    # TRAINING
    train_forward_results: list[ForwardResult] = []
    for i in range(p.n_samples_train):
        u_arr, y_arr, t_arr, u_t = dataset_train[i]

        # 訓練データのQRC状態系列を計算
        solver.reset_rho()
        t_arr_out, states = get_time_evol_states(
            p.n_mpx, u_t, t_arr, solver,
            n_washout=p.n_washout,
            tqdm_title=f"train #{i}" if show_progress else None,
        )

        mask = np.isin(t_arr, t_arr_out)
        t_arr = t_arr[mask]
        u_arr = u_arr[mask]
        y_arr = y_arr[mask]

        train_forward_results.append(
            ForwardResult(
                t=t_arr,
                u=u_arr[:, None],
                states=states,
                y=y_arr[:, None],
            )
        )

    # REGRESSION
    state_all, y_all = [], []
    for i in range(p.n_samples_train):
        state_all.append(np.array(train_forward_results[i].states))
        y_all.append(train_forward_results[i].y)
    state_all = np.concatenate(state_all, axis=0)  # (T * n_samples, n_states)
    y_all = np.concatenate(y_all, axis=0)  # (T * n_samples, n_out)

    lr_model = LinearRegression()
    lr_model.fit(state_all, y_all)

    train_regression_results: list[RegressionResult] = []
    for i in range(p.n_samples_train):
        forward_result = train_forward_results[i]
        y_pred = lr_model.predict(np.array(forward_result.states))
        r2_score = lr_model.score(np.array(forward_result.states), forward_result.y)
        train_regression_results.append(
            RegressionResult(
                t=forward_result.t,
                u=forward_result.u,
                states=forward_result.states,
                y=forward_result.y,
                y_pred=y_pred,
                r2_score=r2_score,
            )
        )

    # TESTING
    test_forward_results: list[ForwardResult] = []
    for i in range(p.n_samples_test):
        u_arr, y_arr, t_arr, u_t = dataset_test[i]

        # テストデータでは新しい状態から開始
        solver.reset_rho()
        t_arr_out, states = get_time_evol_states(
            p.n_mpx, u_t, t_arr, solver,
            n_washout=p.n_washout,
            tqdm_title=f"test #{i}" if show_progress else None,
        )
        mask = np.isin(t_arr, t_arr_out)
        t_arr = t_arr[mask]
        u_arr = u_arr[mask]
        y_arr = y_arr[mask]

        test_forward_results.append(
            ForwardResult(
                t=t_arr,
                u=u_arr[:, None],
                states=states,
                y=y_arr[:, None],
            )
        )

    test_regression_results: list[RegressionResult] = []
    for i in range(p.n_samples_test):
        forward_result = test_forward_results[i]
        y_pred = lr_model.predict(np.array(forward_result.states))
        r2_score = lr_model.score(np.array(forward_result.states), forward_result.y)
        test_regression_results.append(
            RegressionResult(
                t=forward_result.t,
                u=forward_result.u,
                states=forward_result.states,
                y=forward_result.y,
                y_pred=y_pred,
                r2_score=r2_score,
            )
        )

    # RETURN RESULTS
    return QRCExperimentResult(
        param=p,
        train=QRCExperimentResultEntry(
            name=f"train",
            t=np.array([r.t for r in train_regression_results]),
            u=np.array([r.u for r in train_regression_results]),
            states=[r.states for r in train_regression_results],
            y_pred=np.array([r.y_pred for r in train_regression_results]),
            y_true=np.array([r.y for r in train_regression_results]),
            r2_score=np.array([r.r2_score for r in train_regression_results]),
        ),
        test=QRCExperimentResultEntry(
            name=f"test",
            t=np.array([r.t for r in test_regression_results]),
            u=np.array([r.u for r in test_regression_results]),
            states=[r.states for r in test_regression_results],
            y_pred=np.array([r.y_pred for r in test_regression_results]),
            y_true=np.array([r.y for r in test_regression_results]),
            r2_score=np.array([r.r2_score for r in test_regression_results]),
        ),
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
    # searcher = GAParameterSearcher[QRCParam](
    #     scorer=scorer_fn,
    #     param_grid=dict(
    #         n_qubits=[4],
    #         gamma_z=[0.1, 0.01, 0.001, 0.0001, 0.00001],
    #         n_mpx=[1, 2, 4, 6, 8, 10],
    #         j_mean=np.arange(0.1, 2.0, 0.1),
    #         h_mean=np.arange(0.1, 2.0, 0.1),
    #         n_steps=[150],
    #         t_max=[5],
    #         test_ratio=[0.5],
    #         n_washout=[20],
    #         n_samples_train=[8],
    #         n_samples_test=[4],
    #         seed=[0],
    #     ),
    #     param_mapper=param_mapper_fn,
    # )
    # best_param, best_score = searcher.search(n_workers=6)
    #
    # print(f"Best param: {best_param} with R^2={best_score}")
    # df = pd.DataFrame([{**asdict(p), "_score": score} for p, score in searcher.history])
    # df.to_csv('./qrc_param_search_results.csv', index=False)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(df)

    # 最良パラメータで実験・グラフ表示
    best_param = QRCParam(
        n_qubits=4,
        gamma_z=0.001,
        n_mpx=4,
        j_mean=1.0,
        j_std=0.3,
        h_mean=1.0,
        h_std=0.3,
        n_steps=150,
        t_max=5,
        test_ratio=0.5,
        n_washout=20,
        n_samples_train=8,
        n_samples_test=4,
        seed=0,
    )
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
