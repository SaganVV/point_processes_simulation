import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def num_of_neighbors(config: np.ndarray, R: float) -> np.ndarray:
    n = config.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    R_2 = R**2
    for i in prange(n):
        c = 0
        for j in range(n):
            c += np.sum((config[i] - config[j]) ** 2) < R_2
        counts[i] = c - 1
    return counts


@njit(cache=True)
def total_num_of_neighbors(config: np.ndarray, R: float) -> int:
    return np.sum(num_of_neighbors(config, R)) // 2


@njit(cache=True)
def saturated_interaction_statistic(
    config: np.ndarray, R: float, saturation_parameter: float
) -> int:
    neighbors = num_of_neighbors(config, R)
    return np.sum(np.minimum(neighbors, saturation_parameter))
