import numpy as np

from .base import PointProcessDensity
from .utils import total_num_of_neighbors

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