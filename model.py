from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache

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
    def __array__(self) -> np.ndarray:  # (ステップ数, 状態数)
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
    def __array__(self) -> np.ndarray:  # (ステップ数, 状態数)
        # 全てのステップを2次元配列にまとめる
        a = np.array([step.state_vector for step in self._steps])
        assert len(a) == len(self)
        a.setflags(write=False)
        return a


@dataclass(slots=True)
class QRCExperimentResultEntry:
    name: str
    t: np.ndarray  # (T,)
    u: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length is T
    y_pred: np.ndarray  # (T, n_out)
    y_true: np.ndarray | None  # (T, n_out)
    r2_score: float | None

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t, np.ndarray), (type(self.t), self.t)
        assert self.t.ndim == 1, self.t.shape

        assert isinstance(self.u, np.ndarray), (type(self.u), self.u)
        assert self.u.ndim == 2, self.u.shape

        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert len(self.states) == len(self.t), (len(self.states), len(self.t))

        if self.y_true is not None:
            assert isinstance(self.y_true, np.ndarray), (type(self.y_true), self.y_true)
            assert self.y_true.ndim == 2, self.y_true.shape

        assert isinstance(self.y_pred, np.ndarray), (type(self.y_pred), self.y_pred)
        assert self.y_pred.ndim == 2, self.y_pred.shape

        assert len(self.u) == len(self.t), (len(self.u), len(self.t))
        assert len(self.y_pred) == len(self.t), (len(self.y_pred), len(self.t))
        if self.y_true is not None:
            assert len(self.y_true) == len(self.t), (len(self.y_true), len(self.t))
            assert self.y_pred.shape == self.y_true.shape, (self.y_pred.shape, self.y_true.shape)

    def plot_state_series(self, ax=None) -> bool:
        if ax is None:
            ax = plt.gca()
        for i in range(self.states.n_states):
            ax.plot(self.t, self.states.series(i), label=f"State #{i}")
        ax.set_title("Train states time series")
        ax.set_xlabel("Time")
        ax.set_ylabel("State value")
        ax.legend()
        return True

    def plot_prediction_time_series(self, ax=None) -> bool:
        if self.y_true is None:
            return False
        if ax is None:
            ax = plt.gca()
        ax.plot(self.t, self.y_true, label='y (true)')
        ax.plot(self.t, self.y_pred, label='y (pred)', linestyle='--')
        ax.set_title(f'Prediction (R^2={self.r2_score:.3f})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Output')
        ax.legend()
        return True

    @property
    def y_min(self) -> float:
        return min(self.y_true.min(), self.y_pred.min())

    @property
    def y_max(self) -> float:
        return max(self.y_true.max(), self.y_pred.max())

    def plot_prediction_vs_ground_truth(self, ax=None) -> bool:
        if self.y_true is None:
            return False
        ax.scatter(self.y_true, self.y_pred, alpha=0.5)
        ax.plot([self.y_min, self.y_max], [self.y_min, self.y_max], 'k--', label='y=true')
        ax.set_xlabel('y_true')
        ax.set_ylabel('y_pred')
        ax.set_title(f'R²={self.r2_score:.3f}')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return True


@dataclass(frozen=True)
class QRCParam:
    n_qubits: int
    gamma_z: float
    n_mpx: int  # Multiplexing factor
    j_mean: float
    j_std: float
    h_mean: float
    h_std: float
    n_steps: int
    t_max: float
    test_ratio: float
    n_washout: int
    n_lr_train_washout: int
    seed: int

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
        assert isinstance(self.t_max, float), \
            (f"invalid value of `t_max`", type(self.t_max), self.t_max)
        assert isinstance(self.test_ratio, float), \
            (f"invalid value of `test_ratio`", type(self.test_ratio), self.test_ratio)
        assert isinstance(self.n_washout, int), \
            (f"invalid value of `n_washout`", type(self.n_washout), self.n_washout)
        assert isinstance(self.n_lr_train_washout, int), \
            (f"invalid value of `n_lr_train_washout`", type(self.n_lr_train_washout),
             self.n_lr_train_washout)
        assert isinstance(self.seed, int), \
            (f"invalid value of `seed`", type(self.seed), self.seed)


@dataclass
class QRCExperimentResult:
    param: QRCParam
    train: QRCExperimentResultEntry
    test: QRCExperimentResultEntry

    def plot_state_series(self):
        plt.figure(figsize=(14, 6))
        self.train.plot_state_series()
        plt.tight_layout()
        plt.show()

    def plot_prediction(self):
        fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=False,
                                gridspec_kw={'width_ratios': [7, 2]})

        # Plot train data
        self.train.plot_prediction_time_series(ax=axs[0, 0])
        self.train.plot_prediction_vs_ground_truth(ax=axs[0, 1])

        # Plot test data
        self.test.plot_prediction_time_series(ax=axs[1, 0])
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
