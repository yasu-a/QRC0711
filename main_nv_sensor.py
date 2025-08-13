import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.fullstate import fullstate
from core.seed_or_rng import check_seed_or_rng_and_get_rng
from core.time_evol_solver_v2 import create_time_evol_solver
from model.axis import Axis
from model.physical_system import TotalMagnetizationObservable, NoInteractionNVSystem


class NVSensor:
    @classmethod
    def _create_solver(cls, n_qubit: int):
        system = NoInteractionNVSystem.create_instance_with_constant_coefficient(
            value=1.0,
            axis=Axis.Y,
            n_qubit=n_qubit,
        )

        solver = create_time_evol_solver(
            system=system,
            collapse_operator=None,
            init_psi=fullstate(",".join(["z+"] * n_qubit)),
        )

        return solver

    @classmethod
    def _create_observable(cls, n_qubit: int):
        return TotalMagnetizationObservable(
            n_qubit=n_qubit,
            axis=Axis.X,
            normalize=True,
        )

    def __init__(self, *, n_qubit: int = 1, obs_scale: float = 1.0):
        self._n_qubit = n_qubit
        self._obs_scale = obs_scale

    def measure_and_get_expect(self, *, t_begin, t_end, u_t) -> float:
        solver = self._create_solver(self._n_qubit)
        observable = self._create_observable(self._n_qubit)

        t_arr = np.linspace(t_begin, t_end, 21)

        # V2ソルバーで時間発展を計算
        solver.reset_state()
        result = solver.forward(u_t, t_arr)

        # オブザーバブルの期待値を取得
        obs_qobj = observable.create_hamiltonian()[0]
        expect_values = result.expect(obs_qobj)

        return expect_values[-1]  # 最後の時刻の期待値

    def measure_and_get_samples(self, *, t_begin, t_end, u_t, n_samples=1, seed=None, rng=None) \
            -> list[float]:
        """観測値のサンプルを取得して返す"""
        solver = self._create_solver(self._n_qubit)
        observable = self._create_observable(self._n_qubit)

        t_arr = np.linspace(t_begin, t_end, 21)

        # V2ソルバーで時間発展を計算
        solver.reset_state()
        result = solver.forward(u_t, t_arr)

        # 最後の時刻でのStepResultから観測値をサンプリング
        obs_qobj = observable.create_hamiltonian()[0]

        # 乱数生成器を作成
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        # 観測値をサンプリング
        samples = result.sample_measurements(
            obs_qobj, n_samples=n_samples, rng=rng)

        return samples.tolist()


def _test_samples(*, n_qubit: int = 1):
    """サンプルプロット：各b値に対して100個のサンプルを取得してバイオリンプロットで表示"""
    sensor = NVSensor(n_qubit=n_qubit)

    b_arr = np.linspace(-1, 1, 21)  # バイオリンプロット用に点数を調整

    print("Collecting samples for each b value...")

    all_expect = []
    all_samples = []
    all_std = []

    for b in tqdm(b_arr):
        # 期待値を計算
        expect = sensor.measure_and_get_expect(
            t_begin=0,
            t_end=1,
            u_t=lambda t: b,
        )
        all_expect.append(expect)

        # 各b値に対して1000個のサンプルを取得
        samples = sensor.measure_and_get_samples(
            t_begin=0,
            t_end=1,
            u_t=lambda t: b,
            n_samples=1000
        )
        all_samples.append(samples)

        # サンプルから標準偏差を計算
        all_std.append(np.std(samples))

    plt.figure(figsize=(12, 6))

    # 期待値の折れ線グラフをエラーバー付きで描画（エラーバーの線だけ太くする）
    plt.errorbar(
        b_arr, all_expect, yerr=all_std, color='k', marker='o',
        linewidth=3, markersize=0, capsize=0, capthick=0,  # capthick=3でエラーバー線を太く
        label='Expectation value ± std', alpha=0.7
    )

    # バイオリンプロットを作成（平均線の色を赤に設定）＋サンプル点を重ねて描画
    parts = plt.violinplot(all_samples, positions=b_arr,
                           widths=0.08, showmeans=True)
    parts['cmeans'].set_color('red')
    sample_set = set()
    for b, samples in zip(b_arr, all_samples):
        for s in set(samples):
            sample_set.add((float(b), float(s)))
    sample_b, sample_e = zip(*sample_set)
    plt.scatter(sample_b, sample_e, color="blue",
                marker="_", s=40, label='Samples')

    # 表示
    plt.xlabel('B field', fontsize=12)
    plt.ylabel('Measurement samples', fontsize=12)
    plt.title(
        f'NV Sensor: Sample Distribution vs B field\n(Violin plot showing 1000 samples per B value, n_qubit={n_qubit})',
        fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


_test_samples(n_qubit=6)
