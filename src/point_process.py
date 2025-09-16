import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True)
def num_of_neighbors(config : np.ndarray, R:float) -> np.ndarray:
    n = config.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    R_2 = R ** 2
    for i in prange(n):
        c = 0
        for j in range(n):
            c += np.sum((config[i] - config[j]) ** 2) < R_2
        counts[i] = c-1
    return counts

@njit(cache=True)
def total_num_of_neighbors(config : np.ndarray, R:float) -> int:
    return np.sum(num_of_neighbors(config, R)) // 2

@njit(cache=True)
def saturated_interaction_statistic(config: np.ndarray, R:float, saturation_parameter:float) -> int:
    neighbors = num_of_neighbors(config, R)
    return np.sum(np.minimum(neighbors, saturation_parameter))

class PointProcessDensity:

    def __call__(self, config) -> float:
        return np.exp(self.log_density(config))

    def log_density(self, config) -> float:
        raise NotImplementedError("Log density not implemented")

    def log_parangelou(self, config, new_point) -> float:
        new_config = np.vstack([config, new_point])
        return self.log_density(new_config) - self.log_density(config)

    def parangelou(self, config, new_point) -> float:
        return np.exp(self.log_parangelou(config, new_point))

    def log_mixed_parangelou(self, config:np.ndarray, idx_to_remove, new_point):
        new_config = config.copy()
        new_config[idx_to_remove] = new_point
        return self.log_density(new_config) - self.log_density(config)

    def mixed_parangelou(self, config, idx_to_remove, new_point):
        return np.exp(self.log_mixed_parangelou(config, idx_to_remove, new_point))

class PoissonDensity(PointProcessDensity):
    def __init__(self, beta):
        if beta <= 0:
            raise ValueError("Beta must be positive")
        self.log_beta = np.log(beta)
        self.beta = beta

    def log_density(self, config: np.ndarray) -> float:
        return self.log_beta * len(config)

    def log_parangelou(self, config, new_point):
        return self.log_beta

    def parangelou(self, config, new_point):
        return self.beta

    def __repr__(self):
        return f"{self.__class__.__name__}(beta={self.beta})"


class StraussDensity(PointProcessDensity):

    def __init__(self, R, beta, gamma):
        self.R = R
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if not (0 < gamma < 1):
            raise ValueError("Gamma must be in range (0, 1)")

        self.log_beta = np.log(beta)
        self.beta = beta
        self.log_gamma = np.log(gamma)
        self.gamma = gamma

    def _interaction_statistic(self, config):
        return total_num_of_neighbors(config, self.R)

    def log_density(self, config):
        n = len(config)

        res = self.log_beta * n
        inter = self._interaction_statistic(config)

        return res + self.log_gamma * inter

    def log_parangelou(self, config, new_point):
        if config.size == 0:
            return self.log_beta
        dist = np.sum((config - new_point)**2, axis=1)
        return self.log_beta + self.log_gamma * np.sum(dist < self.R**2)

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={self.beta},R={self.R},gamma={self.gamma})"""

class SaturatedDensity(PointProcessDensity):

    def __init__(self, R, beta, gamma, saturation):
        self.R_2 = R ** 2
        self.R = R
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        self.log_beta = np.log(beta)
        self.beta = beta
        self.log_gamma = np.log(gamma)
        self.gamma = gamma
        self.saturation = saturation

    def _interaction_statistic(self, config):
        return saturated_interaction_statistic(config, self.R, self.saturation)

    def log_density(self, config) -> float:
        n = len(config)

        res = self.log_beta * n
        inter = self._interaction_statistic(config)

        second_order_log = self.log_gamma * inter
        return res + second_order_log

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={self.beta},R={self.R},gamma={self.gamma},saturation={self.saturation})"""


class HardcoreDensity(PointProcessDensity):
    def __init__(self, beta, R):
        self.log_beta = np.log(beta)
        self.R = R

    def log_density(self, config):
        inter = total_num_of_neighbors(config, self.R)
        if inter > 0:
            return -np.inf
        return self.log_beta * len(config)

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={np.exp(self.log_beta)}, R={self.R})"""

if __name__ == "__main__":
    R = 0.1
    beta = 100
    gamma = 0.5
    # saturation = 1
    strauss = StraussDensity(R, beta, gamma)
    config = np.array([[0, 0], [1, 0], [0, 1]])
    print(strauss(np.empty(shape=(0, 2))))
    # print(strauss._interaction_statistic(config))
    print(strauss(config))
