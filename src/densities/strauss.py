import numpy as np

from .base import PointProcessDensity
from .utils import total_num_of_neighbors


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
        dist = np.sum((config - new_point) ** 2, axis=1)
        return self.log_beta + self.log_gamma * np.sum(dist < self.R**2)

    def __repr__(self):
        return f"""{self.__class__.__name__}(beta={self.beta},R={self.R},gamma={self.gamma})"""
