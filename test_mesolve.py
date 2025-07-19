# from typing import Callable
# 
# import numpy as np
# import qutip
# 
# from utils import fullgate
# 
# n_qubit = 4
# gamma_z = 0.00001
# delta_t = 0.001
# mesolve_n_step = 100
# n_time_plex = 10
# assert mesolve_n_step % n_time_plex == 0 and mesolve_n_step >= n_time_plex, (mesolve_n_step,
#                                                                              n_time_plex)
# 
# psi = qutip.tensor(*[qutip.basis(2, 0) + qutip.basis(2, 1) for _ in range(n_qubit)]).unit()
# rho = psi * psi.dag()
# 
# random_state_physics = 0
# rng = np.random.RandomState(random_state_physics)
# j_max = 10
# h_max = 3
# g_j = rng.uniform(low=-j_max, high=j_max, size=(n_qubit, n_qubit))
# g_j[np.tril_indices(n_qubit)] = np.nan
# g_h = rng.uniform(low=0, high=h_max, size=n_qubit)
# 
# print(g_j)
# print(g_h)
# 
# c_ops = [gamma_z ** .5 * fullgate(n_qubit, f"Z{i}") for i in range(n_qubit)]
# e_ops = [fullgate(n_qubit, f"X{i}") for i in range(n_qubit)]
# 
# 
# def create_time_dependent_hamiltonian(u_t: Callable[[float], float]):
#     ham_lst = []  # qutip time-dependent hamiltonian
#     # 相互作用
#     for i in range(n_qubit - 1):
#         for j in range(i + 1, n_qubit):
#             ham_lst.append(-g_j[i, j] * fullgate(n_qubit, f"Z{i}, Z{j}"))
#     # 磁場
#     for i in range(n_qubit):
#         ham_lst.append([-g_h[i] * fullgate(n_qubit, f"X{i}"), u_t])
#     return ham_lst
# 
# 
# n_train = 3000
# t_train = np.arange(n_train) * delta_t
# u_train = input_signal[1000:][:n_train]
# y_train = narma10_output[1000:][:n_train]
# 
# 
# def u_t(t: float) -> float:
#     return float(u_train[int(t // delta_t)])
# 
# 
# ham = create_time_dependent_hamiltonian(u_t)
# 
# 
# def create_time_dependent_hamiltonian(u_t: Callable[[float], float]):
#     ham_lst = []  # qutip time-dependent hamiltonian
#     # 相互作用
#     for i in range(n_qubit - 1):
#         for j in range(i + 1, n_qubit):
#             ham_lst.append(-g_j[i, j] * fullgate(n_qubit, f"Z{i}, Z{j}"))
#     # 磁場
#     for i in range(n_qubit):
#         ham_lst.append([-g_h[i] * fullgate(n_qubit, f"X{i}"), u_t])
#     return ham_lst
