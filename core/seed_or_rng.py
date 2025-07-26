import numpy as np


def check_seed_or_rng_and_get_rng(
        *,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
) -> np.random.RandomState:
    if seed is None and rng is None:
        # both seed and rng are ungiven
        rng = np.random.RandomState(0)
    elif seed is None and rng is not None:
        # seed is ungiven but rng is given
        pass
    elif seed is not None and rng is None:
        # seed is given but rng is ungiven
        rng = np.random.RandomState(seed)
    else:
        # both seed and rng are given
        raise ValueError("Only one parameter should be provided: either 'seed' or 'rng', not both")
    return rng
