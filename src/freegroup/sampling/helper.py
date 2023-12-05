import numpy as np

def get_rng(rng = None):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, int):
        return np.random.default_rng(seed = rng)
    return rng
