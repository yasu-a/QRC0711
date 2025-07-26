from dataclasses import dataclass


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
