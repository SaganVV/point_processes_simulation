import numpy as np
from .base import PointProcessDensity

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
