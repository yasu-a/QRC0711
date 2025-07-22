import os
from dataclasses import asdict
from datetime import datetime
from typing import Literal

import numpy as np

from experiment import PredictionResultSet, NVQRCEstimator, PredictionExperimentSuite
from model import NVQRCParam
from service.visualize import plot_state_series, plot_prediction
from utils.dataset_v2 import NoisyDelayedSineDatasetGenerator, DelayedRandomDatasetGenerator

"""
やること
 - 物理系なので横軸をtに統一したい
 - これはFN-QRCではないのでtime-multiplexingの概念にとらわれる必要はない

MPX1単位の相互作用時間はt_delta/n_mpx
シミュレーション最大時間はt_delta*n_stepsで決まる
初期washoutはn_washout
離散時間データセットは1ステップの継続時間t_deltaからn_stepsまでを生成
"""


def run_qrc_experiment(
        param: NVQRCParam,
        *,
        n_steps: int,
        t_max: float,
        n_washout: int,
        n_samples_train: int,
        n_samples_test: int,
        func_type: Literal["lagged_sine", "lagged_random_uniform"],
        seed: int,
        show_progress=False
) -> tuple[PredictionResultSet, PredictionResultSet]:
    if func_type == "lagged_sine":
        def generator_fn(rng: np.random.RandomState):
            return NoisyDelayedSineDatasetGenerator(
                t_max=t_max,
                t_step=t_max / n_steps,
                freq=1.0,
                phase_offset=0.0,
                discrete_lag=5 * param.n_mpx,
                amplitude=1.0,
                noise_std=0.01,
                rng=rng,
            )
    elif func_type == "lagged_random_uniform":
        def generator_fn(rng: np.random.RandomState):
            return DelayedRandomDatasetGenerator(
                t_max=t_max,
                t_step=t_max / n_steps,
                discrete_lag=2,
                low=0.0,
                high=1.0,
                rng=rng,
            )
    else:
        raise ValueError(f"Invalid func_type: {func_type}")

    rng = np.random.RandomState(seed=seed)
    suite = PredictionExperimentSuite(
        generator_fn=generator_fn,
        n_train_samples=n_samples_train,
        n_test_samples=n_samples_test,
        rng=rng,
        model_class=NVQRCEstimator,
        model_kwargs={
            'param': param,
            'seed': seed,
            'n_washout': n_washout,
        },
        show_progress=show_progress,
    )
    suite.run()

    return suite.results_train, suite.results_test


def param_mapper_fn(d: dict) -> NVQRCParam:
    return NVQRCParam(
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
    )


def scorer_fn(p: NVQRCParam) -> float:
    results_train, results_test = run_qrc_experiment(
        p,
        n_steps=300,
        t_max=100.0,
        n_samples_train=4,
        n_samples_test=4,
        n_washout=50,
        func_type="lagged_random_uniform",
        seed=0,
        show_progress=False,
    )
    return results_train.aggregated_score("r2") * 0.3 + results_test.aggregated_score("r2") * 0.7


def constraint_predicate_fn(p: NVQRCParam) -> bool:
    return not (
            p.obs_x is False and
            p.obs_y is False and
            p.obs_z is False
    )


def run_search():
    from search.ga import GAParameterSearcher
    param_grid = dict(
        j_mean=[0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        h_mean=[0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        gamma_z=[0.001, 0.0001, 0.00001],
        n_qubits=[2, 3, 4],
        n_mpx=[1, 2, 4, 8, 12, 16, 20],
        obs_x=[False, True],
        obs_y=[False, True],
        obs_z=[False, True],
    )

    searcher = GAParameterSearcher[NVQRCParam](
        scorer=scorer_fn,
        param_grid=param_grid,
        param_mapper=param_mapper_fn,
        constraint_predicate=constraint_predicate_fn,
        n_pop=60,
        n_gen=5,
        crossover_rate=.90,
        crossover_type="uniform",
        tournament_size=5,
    )
    best_param, best_score = searcher.search(n_workers=8)

    print(f"Best param: {best_param} with R^2={best_score}")
    import pandas as pd
    df = pd.DataFrame([{**asdict(p), "_score": score} for p, score in searcher.history])
    df = df.sort_values('_score', ascending=False)
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df)
    os.makedirs('./results', exist_ok=True)
    df.to_csv(f'./results/qrc_param_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
              index=False)

    return best_param


def run_preview(best_param):
    # 最良パラメータで実験・グラフ表示
    train_results, test_results = run_qrc_experiment(
        best_param,
        n_steps=300,
        t_max=10.0,
        n_samples_train=1,
        n_samples_test=1,
        n_washout=50,
        func_type="lagged_sine",
        seed=0,
        show_progress=True,
    )
    plot_state_series(train_results[0])
    plot_prediction([train_results[0], test_results[0]])


def main():
    # best_param = run_search()
    best_param = NVQRCParam(
        n_qubits=3, gamma_z=0.01, n_mpx=3, j_mean=0.6, j_std=0.3, h_mean=0.1,
        h_std=0.05, obs_x=True, obs_y=True, obs_z=False,
    )
    run_preview(best_param)


if __name__ == '__main__':
    main()

r"""
C:\Users\yasuh\PycharmProjects\QRC0711\.venv\Scripts\python.exe C:\Users\yasuh\PycharmProjects\QRC0711\main.py 
Best R2=0.908, param={'N_QUBITS': 6, 'GAMMA_Z': 0.001, 'V': 8, 'J_MEAN': 0.5, 'J_STD': 1.0, 'H_MEAN': 2.5, 'H_STD': 1.75, 'M': 500, 'T': 30, 'test_ratio': 0.3, 'washout': 10, 'random_seed': 0}:  28%|██▊       | 274/972 [8:26:29<783:50:16, 4042.72s/it]

0.5     2.5
/6C2    /6x2
0.0333  0.2083
"""
