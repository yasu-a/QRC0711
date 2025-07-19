import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

from model import QRCExperimentResult, QRCStateTimeStep, QRCStateSeries, QRCExperimentResultEntry, \
    QRCParam
from physical_system import NVReservoirObservable, NVReservoirCollapseOperator, \
    NVReservoirPhysicsSystem
from time_evol_solver import TimeEvolutionSolver
from utils.axis import Axis
from utils.dataset import train_test_split, to_continuous_function, DelayedSine
from utils.fullstate import fullstate


# QRC状態系列の計算関数
def get_time_evol_states(n_mpx, u_arr, t_mpx_arr, solver) -> tuple[np.ndarray, QRCStateSeries]:
    # 時間依存の入力
    u_t = to_continuous_function(u_arr, t_step=np.diff(t_mpx_arr).mean())

    # 各タイムステップについて
    steps = []
    t_arr = []  # u_arrやstatesに直接対応する（time-multiplexingを考慮しない）時刻の配列
    for i in tqdm(range(len(t_mpx_arr) - 2)):  # mesolveが後方の時刻を参照するため少し前で止める
        # ステップの開始時刻から終了時刻まで時間発展させて、各時刻における結果を得る
        t_begin, t_end = t_mpx_arr[i], t_mpx_arr[i + 1]
        t_div = np.linspace(t_begin, t_end, n_mpx + 1)[:-1]  # shape: (n_mpx,)
        result = solver.forward(u_t, t_div)

        # flatten all qubit states at this time step
        states = np.stack([result.expect(j) for j in range(result.n_expect)],
                          axis=1)  # (n_mpx, 状態数)
        steps.append(QRCStateTimeStep(states=states))
        t_arr.append(t_end)

    return np.array(t_arr), QRCStateSeries(steps=steps)


def run_qrc_experiment(p: QRCParam) -> QRCExperimentResult:
    # サイン波＋ノイズの入力系列を生成
    t_mpx_all = np.linspace(0, p.t_max, p.n_steps * p.n_mpx)
    t_all, y_all = DelayedSine.create(t_mpx_all, random_seed=p.seed, lag=5 * p.n_mpx, freq=1.0)

    # 訓練・テストデータへの分割
    (u_train, u_test), (y_train, y_test), (t_mpx_train, t_mpx_test) = train_test_split(
        t_all, y_all, t_mpx_all,
        n_washout=p.n_washout,
        test_ratio=p.test_ratio,
    )

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
        observable=NVReservoirObservable(n_qubit=p.n_qubits),
        collapse_operator=NVReservoirCollapseOperator(n_qubit=p.n_qubits, gamma_z=p.gamma_z),
        init_psi=fullstate(",".join(["x+"] * p.n_qubits)),
    )

    # 訓練データのQRC状態系列を計算
    solver.reset_rho()
    t_train, states_train = get_time_evol_states(p.n_mpx, u_train, t_mpx_train, solver)
    y_train_true = y_train[:len(states_train)]

    # テストデータのQRC状態系列を計算
    solver.reset_rho()
    t_test, states_test = get_time_evol_states(p.n_mpx, u_test, t_mpx_test, solver)
    y_test_true = y_test[:len(states_test)]

    # 線形回帰による出力予測
    lr_model = LinearRegression()
    lr_model.fit(
        np.array(states_train)[p.n_lr_train_washout:, :],
        y_train_true[p.n_lr_train_washout:],
    )
    y_train_pred = lr_model.predict(np.array(states_train))
    y_test_pred = lr_model.predict(np.array(states_test))
    r2_train = r2_score(y_train_true, y_train_pred)
    r2_test = r2_score(y_test_true, y_test_pred)

    # 結果をデータクラスで返す
    return QRCExperimentResult(
        param=p,
        train=QRCExperimentResultEntry(
            name="train",
            t=t_train,
            u=u_train,
            states=states_train,
            y_true=y_train_true,
            y_pred=y_train_pred,
            r2_score=r2_train,
        ),
        test=QRCExperimentResultEntry(
            name="test",
            t=t_test,
            u=u_test,
            states=states_test,
            y_true=y_test_true,
            y_pred=y_test_pred,
            r2_score=r2_test,
        ),
    )


def main():
    # # パラメータ探索グリッド
    # param_grid = {
    #     'n_qubits': [6],
    #     'gamma_z': [0.001, 0.0001, 0.00001],
    #     'n_mpx': [1, 2, 4, 8],
    #     'j_mean': [0.5, 1.0, 5.0],
    #     'j_std': [0.5, 1.0, 2.0],
    #     'h_mean': [0.25, 0.5, 2.5],
    #     'h_std': [0.175, 0.25, 1.75],
    #     'n_steps': [500],
    #     't_max': [30],
    #     'test_ratio': [0.3],
    #     'n_washout': [10],
    #     'n_lr_train_washout': [20],
    #     'seed': [0],
    # }
    # searcher = AbstractParameterSearcher[QRCParam](
    #     scorer=lambda p: run_qrc_experiment(p).test.r2_score,
    #     param_grid=param_grid,
    #     param_mapper=lambda d: QRCParam(**d),
    # )
    # best_param, best_score = searcher.search(n_workers=4)
    #
    # print(f"Best param: {best_param} with R^2={best_score}")
    # df = pd.DataFrame([{**asdict(p), "_score": score} for p, score in searcher.history])
    # df.to_csv('./qrc_param_search_results.csv', index=False)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(df)

    # 最良パラメータで実験・グラフ表示
    best_param = QRCParam(
        n_qubits=6,
        gamma_z=0.001,
        n_mpx=4,
        j_mean=0.5,
        j_std=1.0,
        h_mean=2.5,
        h_std=1.75,
        n_steps=300,
        t_max=30,
        test_ratio=0.3,
        n_washout=30,
        n_lr_train_washout=30,
        seed=0,
    )
    results = run_qrc_experiment(best_param)
    results.plot_state_series()
    results.plot_prediction()


if __name__ == '__main__':
    main()

r"""
C:\Users\yasuh\PycharmProjects\QRC0711\.venv\Scripts\python.exe C:\Users\yasuh\PycharmProjects\QRC0711\main.py 
Best R2=0.908, param={'N_QUBITS': 6, 'GAMMA_Z': 0.001, 'V': 8, 'J_MEAN': 0.5, 'J_STD': 1.0, 'H_MEAN': 2.5, 'H_STD': 1.75, 'M': 500, 'T': 30, 'test_ratio': 0.3, 'washout': 10, 'random_seed': 0}:  28%|██▊       | 274/972 [8:26:29<783:50:16, 4042.72s/it]
"""
