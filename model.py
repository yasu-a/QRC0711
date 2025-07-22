from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache

import numpy as np


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


@dataclass(slots=True)
class NVQRCParam:
    n_qubits: int
    n_mpx: int
    j_mean: float
    j_std: float
    h_mean: float
    h_std: float
    gamma_z: float
    obs_x: bool
    obs_y: bool
    obs_z: bool

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.n_qubits, int), \
            (f"invalid value of `n_qubits`", type(self.n_qubits), self.n_qubits)
        assert isinstance(self.n_mpx, int), \
            (f"invalid value of `n_mpx`", type(self.n_mpx), self.n_mpx)
        assert isinstance(self.gamma_z, float), \
            (f"invalid value of `gamma_z`", type(self.gamma_z), self.gamma_z)
        assert isinstance(self.j_mean, float), \
            (f"invalid value of `j_mean`", type(self.j_mean), self.j_mean)
        assert isinstance(self.j_std, float), \
            (f"invalid value of `j_std`", type(self.j_std), self.j_std)
        assert isinstance(self.h_mean, float), \
            (f"invalid value of `h_mean`", type(self.h_mean), self.h_mean)
        assert isinstance(self.h_std, float), \
            (f"invalid value of `h_std`", type(self.h_std), self.h_std)
        assert isinstance(self.obs_x, bool), \
            (f"invalid value of `obs_x`", type(self.obs_x), self.obs_x)
        assert isinstance(self.obs_y, bool), \
            (f"invalid value of `obs_y`", type(self.obs_y), self.obs_y)
        assert isinstance(self.obs_z, bool), \
            (f"invalid value of `obs_z`", type(self.obs_z), self.obs_z)


"""
Sin動くようにして
NARMAを動かして
OUNを動かしたい　タイムプレックスはきついので　できればなしがいい
4:2でqubit分けるノイズ
比率の正規化
シミュ：6qubitでxyz全部はかってるけど　→　実際にはxはかる6qubit、yはかる6qubit、zはかる6qubitを用意することになるでしょう
"""
