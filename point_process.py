import numpy as np

def num_of_neighbors(config, R):
    # Two vertices are neighbors if the distance between them is less than or equal to R
    return np.sum(np.linalg.norm(config[:, None, :] - config[None, :, :], axis=-1) < R, axis=1) - 1

def total_num_of_neighbors(config, R):
    return np.sum(num_of_neighbors(config, R)) // 2

def saturated_interaction_statistic(config, R, saturated_parameter):
    neighbors = num_of_neighbors(config, R)
    return np.sum(np.minimum(neighbors, saturated_parameter))

class StraussDensity:

  def __init__(self, R, beta, gamma):
    self.R = R
    self.log_beta = np.log(beta)
    self.log_gamma = np.log(gamma)

  def _interaction_statistic(self, config):
    return total_num_of_neighbors(config, self.R)

  def __call__(self, config):
    '''
    Return density, calculated up to a constant, which is usually unknown
    '''
    n = len(config)

    return np.exp(self.log_beta * n + self.log_gamma * self._interaction_statistic(config))

class SaturatedDensity:

    def __init__(self, R, beta, gamma, saturation):
        self.R = R
        self.log_beta = np.log(beta)
        self.log_gamma = np.log(gamma)
        self.saturation = saturation

    def _interaction_statistic(self, config):
        return saturated_interaction_statistic(config, self.R, self.saturation)

    def __call__(self, config):
        n = len(config)
        first_order_log = self.log_beta * n
        second_order_log = self.log_gamma * self._interaction_statistic(config)
        return np.exp(first_order_log + second_order_log)

if __name__ == "__main__":
    R = 1.1
    beta = 0.1
    gamma = 0.9
    saturation = 1
    strauss = SaturatedDensity(R, beta, gamma, saturation)
    config = np.array([[0, 0], [1, 0], [0, 1]])
    print(strauss(np.empty(shape=(0, 2))))
    print(strauss._interaction_statistic(config))
    print(strauss(config))
