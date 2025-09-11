import numpy as np

def num_of_neighbors(config, R):
    # Two vertices are neighbors if the distance between them is less than R
    return np.sum(np.linalg.norm(config[:, None, :] - config[None, :, :], axis=-1) < R, axis=1) - 1

def total_num_of_neighbors(config, R):
    return np.sum(num_of_neighbors(config, R)) // 2

def saturated_interaction_statistic(config, R, saturation_parameter):
    neighbors = num_of_neighbors(config, R)
    return np.sum(np.minimum(neighbors, saturation_parameter))

class PoissonDensity:
    def __init__(self, beta):
        if beta <= 0:
            raise ValueError('Beta must be positive')
        self.log_beta = np.log(beta)

    def __call__(self, config):
        return np.exp(self.log_beta * len(config))

class StraussDensity:

  def __init__(self, R, beta, gamma):
    self.R = R
    if beta <= 0:
        raise ValueError('Beta must be positive')
    if not (0 < gamma < 1):
        raise ValueError('Gamma must be in range (0, 1)')

    self.log_beta = np.log(beta)
    self.log_gamma = np.log(gamma)

  def _interaction_statistic(self, config):
    return total_num_of_neighbors(config, self.R)

  def __call__(self, config):
    '''
    Return density, calculated up to a constant, which is usually unknown
    '''
    n = len(config)

    res = np.exp(self.log_beta * n)
    inter = self._interaction_statistic(config)

    return res * np.exp(self.log_gamma * inter)

class SaturatedDensity:

    def __init__(self, R, beta, gamma, saturation):
        self.R = R
        if beta <= 0:
            raise ValueError('Beta must be positive')
        if gamma <= 0:
            raise ValueError('Gamma must be positive')
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

class HardcoreDensity:
    def __init__(self, beta, R):
        self.log_beta = np.log(beta)
        self.R = R

    def __call__(self, config):
        inter = total_num_of_neighbors(config, self.R)
        if inter > 0:
            return 0
        return np.exp(self.log_beta * len(config))

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