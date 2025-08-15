import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import qutip
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from core.fullstate import fullstate
from core.seed_or_rng import check_seed_or_rng_and_get_rng
from core.time_evol_solver_v2 import create_time_evol_solver, TimeEvolutionResult
from model.axis import Axis
from model.physical_system import TotalMagnetizationObservable, NoInteractionNVSystem


@dataclass  # slots=Trueだとlokyがpickleできない
class RunResult:
    """シミュレーション実行結果を保持し、期待値やサンプルの計算機能を提供するクラス"""
    time_evol_result: TimeEvolutionResult
    obs_qobj: qutip.Qobj

    def get_expect(self) -> float:
        """期待値を取得"""
        expect_values = self.time_evol_result.expect(self.obs_qobj)
        return float(expect_values[-1])  # 最後の時刻の期待値

    def get_std(self) -> float:
        """標準偏差を取得"""
        std_values = self.time_evol_result.std(self.obs_qobj)
        return float(std_values[-1])  # 最後の時刻の標準偏差

    def get_samples(self, *, n_samples: int = 1, seed=None, rng=None) \
            -> np.ndarray:  # shape: (n_samples,)
        """観測値のサンプルを取得"""
        # 乱数生成器を作成
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        # 観測値をサンプリング
        samples = self.time_evol_result.sample_measurements(
            self.obs_qobj, n_samples=n_samples, rng=rng,
        )

        return samples

    def get_sample_mean(self, *, n_samples: int = 1, seed=None, rng=None) -> float:
        samples = self.get_samples(n_samples=n_samples, seed=seed, rng=rng)
        return float(np.mean(samples))


class NVSensor:
    @classmethod
    def _create_solver(cls, n_qubit: int, h: float, t_begin=0.0):
        system = NoInteractionNVSystem.create_instance_with_constant_coefficient(
            value=h,
            axis=Axis.Y,
            n_qubit=n_qubit,
        )

        solver = create_time_evol_solver(
            system=system,
            collapse_operator=None,
            init_psi=fullstate(",".join(["z+"] * n_qubit)),
        )
        solver.reset_state(t=t_begin)

        return solver

    @classmethod
    def _create_observable(cls, n_qubit: int, obs_scaling: float):
        return TotalMagnetizationObservable(
            n_qubit=n_qubit,
            axis=Axis.X,
            normalize=True,
            scaling=obs_scaling,
        )

    def __init__(self, *, n_qubit: int, h: float, obs_scaling: float):
        self._n_qubit = n_qubit
        self._h = h
        self._obs_scaling = obs_scaling

        # オブザーバブルを事前計算して保存
        self._observable = self._create_observable(
            self._n_qubit, self._obs_scaling)
        self._obs_qobj = self._observable.create_hamiltonian()[0]

    def run_with_t_arr(self, *, t_arr, u_t) -> RunResult:
        solver = self._create_solver(self._n_qubit, h=self._h, t_begin=t_arr[0])

        # V2ソルバーで時間発展を計算
        time_evol_result = solver.forward(u_t, t_arr)

        return RunResult(
            time_evol_result=time_evol_result,
            obs_qobj=self._obs_qobj
        )

    def run(self, *, t_begin, t_end, u_t) -> RunResult:
        """シミュレーションを実行してRunResultを返す"""
        t_arr = np.linspace(t_begin, t_end, 11)
        return self.run_with_t_arr(t_arr=t_arr, u_t=u_t)


def create_standard_sensor(n_qubit: int, t_interact: float):
    # 磁場が[-1, 1]の範囲でセンサーの出力が磁場に対応するように調整されたセンサを作成

    # 初期センサ（パラメータ調整用、h=1.0, obs_scaling=1.0で仮設定）
    base_sensor = NVSensor(n_qubit=n_qubit, h=1.0, obs_scaling=1.0)

    # bに対する期待値e(b)が最大となるbを探索するための関数
    def neg_expectation(b):
        # bに対する期待値を計算し、最大化のため符号を反転して返す
        run_result = base_sensor.run(t_begin=0, t_end=t_interact, u_t=lambda t: b)
        expectation = run_result.get_expect()
        return -expectation

    # 0 <= b <= 1 の範囲でneg_expectationを最小化（=期待値を最大化）
    optimize_result = minimize_scalar(neg_expectation, bounds=(0.0, 1.0), method='bounded')
    b_at_max_expect = optimize_result.x

    b_arr = np.linspace(0, 0.2, 50)

    # sin(wb) = 1 となるbに合わせて、期待値e(b)がsin(wb)の形になるよう係数wを計算
    # wb = np.pi / 2 となるbをb_at_max_expectとし、w = np.pi / 2 / b_at_max_expect
    omega = np.pi / 2 / b_at_max_expect
    e_arr_base = np.sin(omega * b_arr)

    # w~0付近の線形領域（sin(wb) ≈ wb）を探索
    e_linear_base = omega * b_arr
    e_err_base = np.abs(e_arr_base - e_linear_base) / (e_linear_base + 1e-15)
    b_linear_max = b_arr[e_err_base <= 0.005].max()
    e_b_linear_max = base_sensor.run(t_begin=0, t_end=t_interact, u_t=lambda t: b_linear_max) \
        .get_expect()

    # -1から1の磁場範囲を計測できるようhを設定
    h_param = b_linear_max

    # 磁場が1のとき期待値が1になるようobs_scalingを設定
    obs_scaling_param = 1.0 / e_b_linear_max

    # 最終的なパラメータでNVSensorを作成して返す
    sensor = NVSensor(n_qubit=n_qubit, h=h_param, obs_scaling=obs_scaling_param)
    return sensor


def _test_samples(*, n_qubit: int, t_interact: float):
    """サンプルプロット：各b値に対して100個のサンプルを取得してバイオリンプロットで表示"""
    sensor = create_standard_sensor(n_qubit=n_qubit, t_interact=t_interact)

    b_arr = np.linspace(-1, 1, 21)  # バイオリンプロット用に点数を調整

    print("Collecting samples for each b value...")

    n_samples_lst = [100, 1000, 10000, 100000]
    results: list[RunResult] = []
    all_expect = []
    all_samples = []
    all_std = []
    all_sample_means = [[] for _ in n_samples_lst]

    for b in tqdm(b_arr):
        # シミュレーションを実行
        result = sensor.run(
            t_begin=0,
            t_end=t_interact,
            u_t=lambda t: b,
        )
        results.append(result)

    rng = np.random.RandomState(seed=0)
    for result in results:
        all_expect.append(result.get_expect())
        all_std.append(result.get_std())
        all_samples.append(result.get_samples(n_samples=100000, rng=rng))
        for i, n_samples in enumerate(n_samples_lst):
            all_sample_means[i].append(result.get_sample_mean(n_samples=n_samples, rng=rng))
    del rng

    # プロットを作成（縦に3つ並べる、x軸はshare）
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=200, sharex=True)

    # 1つ目のグラフ（元のバイオリンプロット＋エラーバー）
    ax = axes[0]

    # 期待値の折れ線グラフをエラーバー付きで描画（エラーバーの線だけ太くする）
    ax.errorbar(
        b_arr, all_expect, yerr=all_std, color='k', marker='o',
        linewidth=3, markersize=0, capsize=0, capthick=0,  # capthick=3でエラーバー線を太く
        label='Expectation value ± std', alpha=0.7
    )

    # バイオリンプロットを作成（平均線の色を赤に設定）＋サンプル点を重ねて描画
    parts = ax.violinplot(all_samples, positions=b_arr,
                          widths=0.08, showmeans=True)
    parts['cmeans'].set_color('red')
    sample_set = set()
    for b, samples in zip(b_arr, all_samples):
        for s in set(samples):
            sample_set.add((float(b), float(s)))
    sample_b, sample_e = zip(*sample_set)
    ax.scatter(sample_b, sample_e, color="blue",
               marker="_", s=40, label='Samples')

    # 期待値の最小値と最大値
    e_min, e_max = np.min(all_expect), np.max(all_expect)
    ax.axhline(e_max, color='k', linestyle='--',
               linewidth=1, label=f"Max: {e_max:.6f}")
    ax.axhline(e_min, color='k', linestyle='--',
               linewidth=1, label=f"Min: {e_min:.6f}")

    # 設定
    ax.set_ylabel('Measurement samples', fontsize=12)
    ax.set_title(
        f'NV Sensor: Sample Distribution vs B field\n(Violin plot showing 1000 samples per B value, n_qubit={n_qubit})',
        fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2つ目のグラフ（1つ目のグラフのコピー、ylimを-1.2～+1.2に設定）
    ax2 = axes[1]

    # 期待値の折れ線グラフをエラーバー付きで描画
    ax2.errorbar(
        b_arr, all_expect, yerr=all_std, color='k', marker='o',
        linewidth=3, markersize=0, capsize=0, capthick=0,
        label='Expectation value ± std', alpha=0.7
    )
    parts2 = ax2.violinplot(all_samples, positions=b_arr,
                            widths=0.08, showmeans=True)
    parts2['cmeans'].set_color('red')
    ax2.scatter(sample_b, sample_e, color="blue",
                marker="_", s=40, label='Samples')
    ax2.axhline(e_max, color='k', linestyle='--',
                linewidth=1, label=f"Max: {e_max:.6f}")
    ax2.axhline(e_min, color='k', linestyle='--',
                linewidth=1, label=f"Min: {e_min:.6f}")
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_ylabel('Measurement samples', fontsize=12)
    ax2.set_title(
        f'NV Sensor: Sample Distribution (ylim=[-1.2, 1.2])\n(n_qubit={n_qubit})',
        fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3つ目のグラフ（期待値といろいろなサンプル数における期待値の推定値をプロット）
    ax3 = axes[2]
    # 真の期待値を黒の点線で描画
    ax3.plot(b_arr, all_expect, label="Expectation value (true)", color='black', linestyle='dashed',
             alpha=0.7)

    # 赤色の鮮やかさをn_samplesが大きいほど強くする
    reds = np.linspace(0.2, 1.0, len(n_samples_lst))  # 0.4(暗い赤)～1.0(鮮やかな赤)
    for idx, n_samples in enumerate(n_samples_lst):
        sample_means = all_sample_means[idx]
        ax3.plot(
            b_arr, sample_means,
            label=f"n_samples={n_samples}",
            color=(reds[idx], 0, 0),  # R, G, B
            linestyle='solid',
            alpha=0.7,
        )
    ax3.set_xlabel('B field', fontsize=12)
    ax3.set_ylabel('Expectation value', fontsize=12)
    ax3.set_title(
        f'NV Sensor: Expectation value vs B field\n(n_qubit={n_qubit})',
        fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 表示
    plt.tight_layout()
    plt.show()


# _test_samples(n_qubit=4, t_interact=1.0)
# _test_samples(n_qubit=4, t_interact=0.1)
# _test_samples(n_qubit=4, t_interact=0.01)

# sys.exit()


@dataclass(slots=True)
class Trajectory:
    t_arr: np.ndarray
    b_arr: np.ndarray
    valid_mask: np.ndarray

    @classmethod
    def create(cls, *, t_begin, t_end, n_step, seed=None, rng=None) -> "Trajectory":
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        t_arr = np.linspace(t_begin, t_end, n_step)
        # mesolveが少し先を参照するのでt_arrの範囲を延ばす
        _t_arr_extend = np.linspace(t_end, t_end * 2, n_step)
        t_arr = np.concatenate([t_arr, _t_arr_extend[1:]])

        valid_mask = t_arr <= t_end

        b_arr = np.where(
            rng.random(len(t_arr)) < 0.5,
            rng.choice([-0.02, 0.02], size=len(t_arr)),
            0,
        )
        b_arr = np.cumsum(b_arr) + 0.5
        assert np.all(b_arr[valid_mask] >= 0)
        assert np.all(b_arr[valid_mask] <= 1)

        return cls(t_arr=t_arr, b_arr=b_arr, valid_mask=valid_mask)

    def __post_init__(self):
        assert self.t_arr.ndim == 1
        assert self.t_arr.shape == self.b_arr.shape
        assert self.t_arr.shape == self.valid_mask.shape

    def to_continuous_function(self) -> Callable[[float], float]:
        def func(t: float) -> float:
            if t < self.t_arr[0]:
                return 0
            if t > self.t_arr[-1]:
                return float(self.b_arr[-1])
            # t_arrの中でt以下の最大のインデックスを探す
            idx = np.searchsorted(self.t_arr, t, side='right') - 1
            return float(self.b_arr[idx])

        return func


def _show_trajectory_sample():
    rng = np.random.RandomState(seed=0)
    plt.figure()
    t_arr = np.linspace(0, 1, 1000)
    for i in range(5):
        trajectory = Trajectory.create(t_begin=0, t_end=1, n_step=100, rng=rng)
        u_t = trajectory.to_continuous_function()
        plt.plot(t_arr, np.vectorize(u_t)(t_arr), label=f"Trajectory {i + 1}")
    plt.legend()
    plt.show()


# _show_trajectory_sample()


SENS_N_QUBIT = 4
SENS_N_SAMPLES = 1000000
RES_N_QUBIT = 4
RES_N_SAMPLES = 1000000
T_END = 1.0
T_SENS = 0.7
T_SENS_INTERVAL = 0.05


def _worker_trajectory(u_t: Callable[[float], float]) -> tuple[RunResult, RunResult]:
    # nv_reservoirの時間発展 [0, T_END]
    nv_reservoir = create_standard_sensor(n_qubit=RES_N_QUBIT, t_interact=T_END)
    result_reservoir = nv_reservoir.run(
        t_begin=0,
        t_end=T_END,
        u_t=u_t,
    )

    # nv_sensの時間発展 [T_SENS - T_SENS_INTERVAL, T_SENS + T_SENS_INTERVAL]
    nv_sens = create_standard_sensor(n_qubit=SENS_N_QUBIT, t_interact=T_SENS_INTERVAL * 2)
    result_sens = nv_sens.run(
        t_begin=T_SENS - T_SENS_INTERVAL,
        t_end=T_SENS + T_SENS_INTERVAL,
        u_t=u_t,
    )

    return result_reservoir, result_sens


@dataclass(slots=True)
class ComputationResult:
    trajectory: Trajectory
    b_reservoir: float
    b_sens: float
    b_true: float


def compute_b(trajectories: list[Trajectory]) -> list[ComputationResult]:
    rng = np.random.RandomState(seed=0)
    u_t_lst = [trajectory.to_continuous_function() for trajectory in trajectories]
    worker_results = list(
        Parallel(n_jobs=os.cpu_count() - 1)(
            delayed(_worker_trajectory)(u_t)
            for u_t in tqdm(u_t_lst)
        )
    )
    computation_results = []
    for trajectory, u_t, (result_reservoir, result_sens) \
            in tqdm(zip(trajectories, u_t_lst, worker_results)):
        b_reservoir = result_reservoir.get_sample_mean(n_samples=RES_N_SAMPLES, rng=rng) * 1
        b_sens = result_sens.get_sample_mean(n_samples=SENS_N_SAMPLES, rng=rng) * 1
        # ^ standard_sensorは測定値と磁場がx1で対応する
        b_true = u_t(T_SENS)
        computation_results.append(
            ComputationResult(
                trajectory=trajectory,
                b_reservoir=b_reservoir,
                b_sens=b_sens,
                b_true=b_true,
            )
        )
    return computation_results


def main():
    rng = np.random.RandomState(seed=0)
    ds = [
        Trajectory.create(t_begin=0, t_end=T_END, n_step=100, rng=rng)
        for _ in range(200)
    ]
    ds_train = ds[:150]
    ds_test = ds[150:]

    b_reservoir_arr = []
    b_sens_arr = []
    b_sens_true_arr = []
    for compute_result in compute_b(ds_train):
        b_reservoir_arr.append(compute_result.b_reservoir)
        b_sens_arr.append(compute_result.b_sens)
        b_sens_true_arr.append(compute_result.b_true)

    # 相関をプロット
    plt.figure(dpi=200)
    plt.scatter(
        b_sens_true_arr,
        b_sens_arr,
        s=5,
        edgecolors='blue',
        facecolors='white',
        alpha=0.5,
        label="Sensor",
    )
    plt.scatter(
        b_reservoir_arr,
        b_sens_arr,
        s=5,
        edgecolors='red',
        facecolors='white',
        alpha=0.5,
        label="Reservoir",
    )
    plt.plot([-2, 2], [-2, 2], "k--")
    plt.xlabel("Reservoir / sensor true")
    plt.ylabel("Sensor")
    plt.title("Correlation between Reservoir and Sensor")
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig("corr.png")
    plt.show()


if __name__ == '__main__':
    main()
