from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

Discrete = np.ndarray | float
Continuous = Callable[[Discrete], Discrete]


@dataclass(frozen=True)
class Dataset:
    name: str
    parameters: dict[str, Any]
    t_seq: Discrete
    u_seq: Discrete
    y_true_seq: Discrete
    u_t: Continuous
    y_true_t: Continuous
