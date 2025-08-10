from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.fullstate import fullstate
from core.time_evol_solver import create_time_evol_solver
from experiment.estimator.nvqrc import NVQRCEstimator
from experiment.suite.prediction import PredictionExperimentSuite
from model.axis import Axis
from model.param import NVQRCParam
from model.physical_system import NVPhysicalSystem, TotalMagnetizationObservable
from model.prediction_result import PredictionResultSet
from service.compute_time_evol import get_compute_time_evol_state_series_service
from service.dataset import NoisyDelayedSineDatasetGenerator, DelayedRandomDatasetGenerator, \
    DelayedSineDiscreteDatasetGenerator

"""
やること
 - 物理系なので横軸をtに統一したい
 - これはFN-QRCではないのでtime-multiplexingの概念にとらわれる必要はない

MPX1単位の相互作用時間はt_delta/n_mpx
シミュレーション最大時間はt_delta*n_stepsで決まる
初期washoutはn_washout
離散時間データセットは1ステップの継続時間t_deltaからn_stepsまでを生成
"""


def create_generator_fn(
        func_type: Literal["lagged_sine", "lagged_random_uniform"],
        t_max: float,
        n_steps: int,
):
    if func_type == "lagged_sine":
        def generator_fn(rng: np.random.RandomState):
            return NoisyDelayedSineDatasetGenerator(
                t_max=t_max,
                t_step=t_max / n_steps,
                freq=1.0,
                phase_offset=rng.uniform(0, 2 * np.pi),
                discrete_lag=5,
                amplitude=1.0,
                noise_std=0.000,
                rng=rng,
            )
    elif func_type == "lagged_sine_discrete":
        def generator_fn(rng: np.random.RandomState):
            return DelayedSineDiscreteDatasetGenerator(
                t_max=t_max,
                t_step=t_max / n_steps,
                freq=1.0,
                phase_offset=rng.uniform(0, 2 * np.pi),
                discrete_lag=5,
                amplitude=1.0,
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
    return generator_fn


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
        show_progress=False,
        n_cpu=-1,
) -> tuple[PredictionResultSet, PredictionResultSet]:
    generator_fn = create_generator_fn(
        func_type=func_type,
        t_max=t_max,
        n_steps=n_steps,
    )
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
            "n_cpu": n_cpu,
        },
        show_progress=show_progress,
    )
    suite.run()

    return suite.results_train, suite.results_test


@dataclass(frozen=True)
class Param:
    n_qubit: int


def main():
    param = Param(
        n_qubit=8,
    )

    system = NVPhysicalSystem(
        n_qubit=param.n_qubit,
        j_mean=1.0,
        j_std=1.0,
        j_axis=[Axis.X, Axis.Y, Axis.Z],
        h_mean=1.0,
        h_std=1.0,
        h_axis=Axis.X,
        h_td_axis=Axis.Z,
        seed=0,
    )

    solver = create_time_evol_solver(
        system=system,
        observable=TotalMagnetizationObservable(
            n_qubit=param.n_qubit,
            axis=Axis.Z,
        ),
        init_psi=fullstate(",".join(["z+"] * param.n_qubit)) + fullstate(
            ",".join(["z-"] * param.n_qubit)),
    )

    b_arr = np.linspace(-100, 100, 100)
    e_arr = []
    for b in tqdm(b_arr):
        t_arr = np.linspace(0, 0.01, 101)
        u_t = lambda t: b

        valid_time_mask, state_series = get_compute_time_evol_state_series_service().execute(
            solver=solver,
            n_mpx=1,
            u_t=u_t,
            t_arr=t_arr,
            reset_state=True,
            full_span=True,
        )

        e_arr.append(state_series[-1, 0])
    e_arr = np.array(e_arr)

    plt.plot(b_arr, e_arr)
    plt.show()


if __name__ == '__main__':
    main()
