from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

Discrete = np.ndarray | float
Continuous = Callable[[Discrete], Discrete]


@dataclass(frozen=True)
class Dataset:
    name: str
    parameters: dict[str, Any]
    t_arr: Discrete
    u_arr: Discrete
    y_arr: Discrete
    u_t: Continuous
    y_t: Continuous
