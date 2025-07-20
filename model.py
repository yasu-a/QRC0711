from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class QRCStateTimeStep:
    def __init__(
            self,
            states: np.ndarray,  # shape: (V, 状態数)
    ):
        assert states.ndim == 2, states.shape
        self._states = states

    def __len__(self) -> int:
        return self._states.shape[0]

    @property
    def n_states_per_times(self) -> int:
        return self._states.shape[1]

    @property
    def n_states(self) -> int:
        return len(self) * self.n_states_per_times

    @property
    def state_vector(self) -> np.ndarray:
        v = self._states.flatten()
        assert len(v) == self.n_states
        return v


class AbstractStateSeries(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def n_states(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # (ステップ数, 状態数)
        # 全てのステップを2次元配列にまとめる
        raise NotImplementedError()

    def series(self, i: int) -> np.ndarray:
        return np.array(self)[:, i]


class QRCStateSeries(AbstractStateSeries):
    def __init__(
            self,
            steps: list[QRCStateTimeStep],
    ):
        self._steps = steps
        assert all(step.n_states == self._steps[0].n_states for step in self._steps[1:])
        self._n_states = self._steps[0].n_states

    def __len__(self) -> int:
        # number of steps
        return len(self._steps)

    @property
    def n_states(self) -> int:
        return self._n_states

    @cache
    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # (ステップ数, 状態数)
        # 全てのステップを2次元配列にまとめる
        a = np.array([step.state_vector for step in self._steps], dtype=dtype, copy=copy)
        assert len(a) == len(self)
        a.setflags(write=False)
        return a

    def __getitem__(self, key: slice | np.ndarray):
        if isinstance(key, slice):
            return type(self)(self._steps[key])
        elif isinstance(key, np.ndarray):
            assert key.dtype == bool, f"Array indexing must use boolean mask array, got {key.dtype}"
            assert len(key) == len(self._steps), \
                f"Boolean mask length {len(key)} does not match steps length {len(self._steps)}"
            return type(self)([step for i, step in enumerate(self._steps) if key[i]])
        else:
            assert False


@dataclass(frozen=True)
class QRCParam:
    n_qubits: int
    gamma_z: float
    n_mpx: int
    j_mean: float
    j_std: float
    h_mean: float
    h_std: float
    n_steps: int
    obs_x: bool
    obs_y: bool
    obs_z: bool
    t_max: float
    n_washout: int
    n_samples_train: int
    n_samples_test: int
    seed: int
    func_type: Literal["lagged_sine", "lagged_random_uniform"]

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.n_qubits, int), \
            (f"invalid value of `n_qubits`", type(self.n_qubits), self.n_qubits)
        assert isinstance(self.gamma_z, float), \
            (f"invalid value of `gamma_z`", type(self.gamma_z), self.gamma_z)
        assert isinstance(self.n_mpx, int), \
            (f"invalid value of `n_mpx`", type(self.n_mpx), self.n_mpx)
        assert isinstance(self.j_mean, float), \
            (f"invalid value of `j_mean`", type(self.j_mean), self.j_mean)
        assert isinstance(self.j_std, float), \
            (f"invalid value of `j_std`", type(self.j_std), self.j_std)
        assert isinstance(self.h_mean, float), \
            (f"invalid value of `h_mean`", type(self.h_mean), self.h_mean)
        assert isinstance(self.h_std, float), \
            (f"invalid value of `h_std`", type(self.h_std), self.h_std)
        assert isinstance(self.n_steps, int), \
            (f"invalid value of `n_steps`", type(self.n_steps), self.n_steps)
        assert isinstance(self.obs_x, bool), \
            (f"invalid value of `obs_x`", type(self.obs_x), self.obs_x)
        assert isinstance(self.obs_y, bool), \
            (f"invalid value of `obs_y`", type(self.obs_y), self.obs_y)
        assert isinstance(self.obs_z, bool), \
            (f"invalid value of `obs_z`", type(self.obs_z), self.obs_z)
        assert isinstance(self.t_max, (float, int)), \
            (f"invalid value of `t_max`", type(self.t_max), self.t_max)
        assert isinstance(self.n_washout, int), \
            (f"invalid value of `n_washout`", type(self.n_washout), self.n_washout)
        assert isinstance(self.n_samples_train, int), \
            (f"invalid value of `n_samples_train`", type(self.n_samples_train),
             self.n_samples_train)
        assert isinstance(self.n_samples_test, int), \
            (f"invalid value of `n_samples_test`", type(self.n_samples_test), self.n_samples_test)
        assert isinstance(self.seed, int), \
            (f"invalid value of `seed`", type(self.seed), self.seed)


@dataclass(slots=True)
class QRCExperimentResultEntry:
    name: str
    t: np.ndarray  # (N, T,)  N: sample count
    u: np.ndarray  # (N, T, n_in)
    states: list[AbstractStateSeries]  # length of list is N and length of series if T
    y_pred: np.ndarray  # (N, T, n_out)
    y_true: np.ndarray | None  # (N, T, n_out)
    r2_score: np.ndarray | None  # (N,)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t, np.ndarray), (type(self.t), self.t)
        assert self.t.ndim == 2, self.t.shape

        assert isinstance(self.u, np.ndarray), (type(self.u), self.u)
        assert self.u.ndim == 3, self.u.shape

        assert isinstance(self.states, list), (type(self.states), self.states)
        assert all(isinstance(state, AbstractStateSeries) for state in self.states), self.states
        assert len(self.states) == self.t.shape[0], (len(self.states), self.t.shape[0])

        if self.y_true is not None:
            assert isinstance(self.y_true, np.ndarray), (type(self.y_true), self.y_true)
            assert self.y_true.ndim == 3, self.y_true.shape

        assert isinstance(self.y_pred, np.ndarray), (type(self.y_pred), self.y_pred)
        assert self.y_pred.ndim == 3, self.y_pred.shape

        # 各サンプルの長さが一致することを確認
        assert self.t.shape[1] == self.u.shape[1], (self.t.shape, self.u.shape)
        assert self.u.shape[1] == self.y_pred.shape[1], (self.u.shape, self.y_pred.shape)
        if self.y_true is not None:
            assert self.y_true.shape[1] == self.y_pred.shape[1], (self.y_true.shape,
                                                                  self.y_pred.shape)

        # 各サンプルの状態系列の長さが一致することを確認
        assert all(len(state) == self.t.shape[1] for state in self.states), \
            f"State series lengths {[len(state) for state in self.states]} do not match time length {self.t.shape[1]}"

        # サンプル数が一致することを確認
        assert self.t.shape[0] == self.u.shape[0], (self.t.shape[0], self.u.shape[0])
        assert self.u.shape[0] == self.y_pred.shape[0], (self.u.shape[0], self.y_pred.shape[0])
        if self.y_true is not None:
            assert self.y_true.shape[0] == self.y_pred.shape[0], (self.y_true.shape[0],
                                                                  self.y_pred.shape[0])

        # r2_scoreの検証
        if self.r2_score is not None:
            assert isinstance(self.r2_score, np.ndarray), (type(self.r2_score), self.r2_score)
            assert self.r2_score.shape[0] == self.t.shape[0], (self.r2_score.shape[0],
                                                               self.t.shape[0])

    @property
    def n_samples(self) -> int:
        return self.t.shape[0]

    @property
    def n_steps(self) -> int:
        return self.t.shape[1]

    @property
    def n_in(self) -> int:
        return self.u.shape[2]

    @property
    def n_out(self) -> int:
        return self.y_pred.shape[2]

    @property
    def y_min(self) -> float:
        y_true_min = self.y_true.min() if self.y_true is not None else np.inf
        return min(y_true_min, self.y_pred.min())

    @property
    def y_max(self) -> float:
        y_true_max = self.y_true.max() if self.y_true is not None else -np.inf
        return max(y_true_max, self.y_pred.max())

    @property
    def r2_score_avg(self) -> float:
        if self.r2_score is None:
            raise ValueError("r2_score is not given")
        return float(np.mean(self.r2_score))

    def plot_state_series(self, ax=None, *, x_label: Literal["time", "index"] = "index",
                          sample_index: int = 0) -> bool:
        if ax is None:
            ax = plt.gca()

        if sample_index >= self.n_samples:
            return False

        x_data = self.t[sample_index] if x_label == "time" else np.arange(self.n_steps)
        states_array = np.array(self.states[sample_index])
        for i in range(states_array.shape[1]):
            ax.plot(x_data, states_array[:, i], label=f"State #{i}", lw=1)
        ax.set_title(f"{self.name} states time series (sample {sample_index})")
        ax.set_xlabel("Time" if x_label == "time" else "Index")
        ax.set_ylabel("State value")
        ax.legend()
        return True

    def plot_prediction_time_series(self, ax=None, *,
                                    x_label: Literal["time", "index"] = "index",
                                    sample_index: int = 0) -> bool:
        if self.y_true is None:
            return False
        if sample_index >= self.n_samples:
            return False
        if ax is None:
            ax = plt.gca()
        x_data = self.t[sample_index] if x_label == "time" else np.arange(self.n_steps)

        # 入力uをプロット
        for i in range(self.n_in):
            ax.plot(x_data, self.u[sample_index, :, i], label=f'u (#{i})', alpha=0.7, linewidth=1,
                    color="gray")

        # 2次元配列の各次元をプロット
        for i in range(self.n_out):
            ax.plot(x_data, self.y_true[sample_index, :, i], label=f'y_true (#{i})', lw=0.7,
                    color="tab:green")
            ax.plot(x_data, self.y_pred[sample_index, :, i], label=f'y_pred (#{i})', lw=0.7,
                    color="tab:red")

        r2_display = self.r2_score[sample_index] if self.r2_score is not None else float("nan")
        ax.set_title(f'{self.name} Prediction (R^2={r2_display:.3f}, sample {sample_index})')
        ax.set_xlabel('Time' if x_label == "time" else 'Index')
        ax.set_ylabel('Output')
        ax.legend()
        return True

    def plot_prediction_vs_ground_truth(self, ax=None) -> bool:
        if self.y_true is None:
            return False
        if ax is None:
            ax = plt.gca()

        # すべてのサンプルについて集計
        for sample_idx in range(self.n_samples):
            for i in range(self.n_out):
                ax.scatter(self.y_true[sample_idx, :, i], self.y_pred[sample_idx, :, i],
                           alpha=0.3, s=2, color=f'C{i}')

        # 有効なr2_scoreの平均を計算
        if self.r2_score is not None:
            avg_r2 = np.mean(self.r2_score)
        else:
            avg_r2 = float("nan")

        ax.plot([self.y_min, self.y_max], [self.y_min, self.y_max], 'k--', label='y_pred = y_true')
        ax.set_xlabel('y_true')
        ax.set_ylabel('y_pred')
        ax.set_title(f'{self.name} R²={avg_r2:.3f} avg')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return True


@dataclass
class QRCExperimentResult:
    param: QRCParam
    train: QRCExperimentResultEntry
    test: QRCExperimentResultEntry

    def plot_state_series(self, *, x_label: Literal["time", "index"] = "index",
                          sample_index: int = 0):
        plt.figure(figsize=(13, 8))
        self.train.plot_state_series(x_label=x_label, sample_index=sample_index)
        plt.tight_layout()
        plt.show()

    def plot_prediction(self, *, x_label: Literal["time", "index"] = "index",
                        sample_index: int = 0):
        fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=False,
                                gridspec_kw={'width_ratios': [7, 2]})

        # Plot train data
        self.train.plot_prediction_time_series(ax=axs[0, 0], x_label=x_label,
                                               sample_index=sample_index)
        self.train.plot_prediction_vs_ground_truth(ax=axs[0, 1])

        # Plot test data
        self.test.plot_prediction_time_series(ax=axs[1, 0], x_label=x_label,
                                              sample_index=sample_index)
        self.test.plot_prediction_vs_ground_truth(ax=axs[1, 1])

        plt.tight_layout()
        plt.show()


"""
Sin動くようにして
NARMAを動かして
OUNを動かしたい　タイムプレックスはきついので　できればなしがいい
4:2でqubit分けるノイズ
比率の正規化
シミュ：6qubitでxyz全部はかってるけど　→　実際にはxはかる6qubit、yはかる6qubit、zはかる6qubitを用意することになるでしょう
"""
