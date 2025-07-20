from dataclasses import asdict
from datetime import datetime
import os

import numpy as np

from experiment import run_qrc_experiment
from model import QRCParam


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
        func_type=d["func_type"],
    )


def scorer_fn(p: QRCParam) -> float:
    return run_qrc_experiment(p).test.r2_score_avg


def constraint_predicate_fn(p: QRCParam) -> bool:
    return not (
        p.obs_x is False and
        p.obs_y is False and
        p.obs_z is False
    )


def main_search():
    from search.ga import GAParameterSearcher
    searcher = GAParameterSearcher[QRCParam](
        scorer=scorer_fn,
        param_grid=dict(
            n_qubits=[4],
            gamma_z=[0.1, 0.01, 0.001, 0.0001, 0.00001],
            n_mpx=[1, 2, 4, 8, 12, 16, 20],
            j_mean=[0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            h_mean=[0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            obs_x=[False, True],
            obs_y=[False, True],
            obs_z=[False, True],
            n_steps=[300],
            t_max=[1, 3, 10, 30, 100],
            test_ratio=[0.5],
            n_washout=[50],
            n_samples_train=[4],
            n_samples_test=[2],
            seed=[0],
            func_type=["lagged_random_uniform"],
        ),
        param_mapper=param_mapper_fn,
        constraint_predicate=constraint_predicate_fn,
    )
    best_param, best_score = searcher.search(n_workers=1)

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


def main_show(best_param):
    # 最良パラメータで実験・グラフ表示
    results = run_qrc_experiment(best_param, show_progress=True)
    results.plot_state_series()
    results.plot_prediction()


def main():
    best_param = main_search()
    # best_param = QRCParam(
    #     n_qubits=4, gamma_z=0.01, n_mpx=3, j_mean=0.6, j_std=0.3, h_mean=0.1,
    #     h_std=0.05, n_steps=300, obs_x=True, obs_y=True, obs_z=False, t_max=5.0,
    #     test_ratio=0.5, n_washout=50, n_samples_train=10, n_samples_test=3,
    #     seed=0, func_type='lagged_random_uniform',
    # )
    main_show(best_param)


if __name__ == '__main__':
    main()

r"""
C:\Users\yasuh\PycharmProjects\QRC0711\.venv\Scripts\python.exe C:\Users\yasuh\PycharmProjects\QRC0711\main.py 
Best R2=0.908, param={'N_QUBITS': 6, 'GAMMA_Z': 0.001, 'V': 8, 'J_MEAN': 0.5, 'J_STD': 1.0, 'H_MEAN': 2.5, 'H_STD': 1.75, 'M': 500, 'T': 30, 'test_ratio': 0.3, 'washout': 10, 'random_seed': 0}:  28%|██▊       | 274/972 [8:26:29<783:50:16, 4042.72s/it]

0.5     2.5
/6C2    /6x2
0.0333  0.2083
"""
