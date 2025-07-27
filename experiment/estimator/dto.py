from dataclasses import dataclass

import numpy as np

from model.state_array import AbstractState2DArray


@dataclass(slots=True)
class StateComputationResult:
    t_seq: np.ndarray  # (T,)
    u_seq_n: np.ndarray  # (T, n_in)
    x_seq_n: AbstractState2DArray  # length: (T, n_state)
    y_seq_n: np.ndarray  # (T, n_out)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t_seq, np.ndarray), (type(self.t_seq), self.t_seq)
        assert isinstance(self.u_seq_n, np.ndarray), (type(self.u_seq_n), self.u_seq_n)
        assert isinstance(self.x_seq_n, AbstractState2DArray), (type(self.x_seq_n), self.x_seq_n)
        assert isinstance(self.y_seq_n, np.ndarray), (type(self.y_seq_n), self.y_seq_n)

        n_t = len(self.t_seq)
        assert self.t_seq.ndim == 1, self.t_seq.shape
        assert self.u_seq_n.ndim == 2, self.u_seq_n.shape
        assert self.u_seq_n.shape[0] == n_t, (self.u_seq_n.shape[0], n_t)
        assert len(self.x_seq_n) == n_t, (len(self.x_seq_n), n_t)
        assert self.y_seq_n.ndim == 2, self.y_seq_n.shape
        assert self.y_seq_n.shape[0] == n_t, (self.y_seq_n.shape[0], n_t)

        # copy arrays and make readonly
        self.t_seq = self.t_seq.copy()
        self.t_seq.setflags(write=False)
        self.u_seq_n = self.u_seq_n.copy()
        self.u_seq_n.setflags(write=False)
        self.y_seq_n = self.y_seq_n.copy()
        self.y_seq_n.setflags(write=False)
