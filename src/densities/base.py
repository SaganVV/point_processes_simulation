import numpy as np

from abc import ABC, abstractmethod


class PointProcessDensity(ABC):

    def __call__(self, config) -> float:
        return np.exp(self.log_density(config))

    @abstractmethod
    def log_density(self, config) -> float:
        raise NotImplementedError("Log density not implemented")

    def log_parangelou(self, config, new_point) -> float:
        new_config = np.vstack([config, new_point])
        return self.log_density(new_config) - self.log_density(config)

    def parangelou(self, config, new_point) -> float:
        return np.exp(self.log_parangelou(config, new_point))

    def log_mixed_parangelou(self, config: np.ndarray, idx_to_remove, new_point):
        new_config = config.copy()
        new_config[idx_to_remove] = new_point
        return self.log_density(new_config) - self.log_density(config)

    def mixed_parangelou(self, config, idx_to_remove, new_point):
        return np.exp(self.log_mixed_parangelou(config, idx_to_remove, new_point))
