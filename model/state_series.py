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

    @abstractmethod
    def __getitem__(self, key: slice | np.ndarray) -> "AbstractStateSeries":
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
