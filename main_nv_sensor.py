from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.fullstate import fullstate
from core.time_evol_solver import create_time_evol_solver
from model.axis import Axis
from model.physical_system import TotalMagnetizationObservable, SingleNVSystem
from service.compute_time_evol import get_compute_time_evol_state_series_service


@dataclass(frozen=True)
class Param:
    n_qubit: int


def main():
    param = Param(
        n_qubit=1,
    )

    system = SingleNVSystem(
        h=1.0,
        axis=Axis.Y,
    )

    solver = create_time_evol_solver(
        system=system,
        observable=TotalMagnetizationObservable(
            n_qubit=param.n_qubit,
            axis=Axis.X,
        ),
        init_psi=fullstate(",".join(["z+"] * param.n_qubit)),
    )

    b_arr = np.linspace(-1, 1, 100)
    e_arr = []
    for b in tqdm(b_arr):
        t_arr = np.linspace(0, 1, 11)
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
    print(e_arr)

    plt.plot(b_arr, e_arr)
    plt.show()


if __name__ == '__main__':
    main()
