import numpy as np
from functools import lru_cache

def num_of_neighbors(config, R):
    # Two vertices are neighbors if the distance between them is less than R
    return (
        np.sum(
            np.linalg.norm(config[:, None, :] - config[None, :, :], axis=-1) < R, axis=1
        )
        - 1
    )

def total_num_of_neighbors(config, R):
    return np.sum(num_of_neighbors(config, R)).item() // 2

def saturated_interaction_statistic(config, R, saturation_parameter):
    neighbors = num_of_neighbors(config, R)
    return np.sum(np.minimum(neighbors, saturation_parameter))

class PointProcessDensity:

    def __call__(self, config):
        ...
    def parangelou(self, config, new_point):
        new_config = np.copy(config)
        new_config = np.concatenate((new_config, new_point), axis=0)
        return self(new_config) / self(config)

class PoissonDensity(PointProcessDensity):
    def __init__(self, beta):
        if beta <= 0:
            raise ValueError("Beta must be positive")
        self.log_beta = np.log(beta)
        self.beta = beta

    def __call__(self, config):
        return np.exp(self.log_beta * len(config))

    def parangelou(self, config, new_point):
        return self.beta

    def __repr__(self):
        return f"{self.__class__.__name__}(beta={np.exp(self.log_beta)})"


class StraussDensity(PointProcessDensity):

    def __init__(self, R, beta, gamma):
        self.R = R
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if not (0 < gamma < 1):
            raise ValueError("Gamma must be in range (0, 1)")

        self.log_beta = np.log(beta)
        self.log_gamma = np.log(gamma)

    def _interaction_statistic(self, config):
        return total_num_of_neighbors(config, self.R)

    def __call__(self, config):
        n = len(config)

        res = np.exp(self.log_beta * n)
        inter = self._interaction_statistic(config)

        return res * np.exp(self.log_gamma * inter)

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={np.exp(self.log_beta)},R={self.R},gamma={np.exp(self.log_gamma)})"""


class SaturatedDensity(PointProcessDensity):

    def __init__(self, R, beta, gamma, saturation):
        self.R = R
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        self.log_beta = np.log(beta)
        self.log_gamma = np.log(gamma)
        self.saturation = saturation

    def _interaction_statistic(self, config):
        return saturated_interaction_statistic(config, self.R, self.saturation)

    def __call__(self, config):
        n = len(config)

        res = np.exp(self.log_beta * n)
        inter = self._interaction_statistic(config)

        second_order_log = self.log_gamma * inter
        return res * np.exp(second_order_log)

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={np.exp(self.log_beta)},R={self.R},gamma={np.exp(self.log_gamma)},saturation={self.saturation})"""


class HardcoreDensity(PointProcessDensity):
    def __init__(self, beta, R):
        self.log_beta = np.log(beta)
        self.R = R

    def __call__(self, config):
        inter = total_num_of_neighbors(config, self.R)
        if inter > 0:
            return 0
        return np.exp(self.log_beta * len(config))

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={np.exp(self.log_beta)}, R={self.R})"""


if __name__ == "__main__":
    R = 1.1
    beta = 0.1
    gamma = 0
    # saturation = 1
    strauss = StraussDensity(R, beta, gamma)
    config = np.array([[0, 0], [1, 0], [0, 1]])
    print(strauss(np.empty(shape=(0, 2))))
    # print(strauss._interaction_statistic(config))
    print(strauss(config))
