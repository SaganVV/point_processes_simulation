import numpy as np

from .base import PointProcessDensity
from .utils import saturated_interaction_statistic


class SaturatedDensity(PointProcessDensity):

    def __init__(self, R, beta, gamma, saturation):
        self.R_2 = R**2
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
