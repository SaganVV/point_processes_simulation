import numpy as np

import warnings
from src.regions import Region, Rectangle

from functools import lru_cache

@lru_cache(maxsize=5)
def uniform_probs(n):
    return np.ones(n) / n

def uniform_density(config):
    return uniform_probs(len(config))

class StaticOrDynamic:
    def __init__(self, value):
        if callable(value):
            self._fn = value
            self._value = None
        else:
            self._fn = None
            self._value = value

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs) if self._fn is not None else self._value


class KernelSampler:
    def __init__(self, rng=None, seed=None):
        self.rng = np.random.default_rng(seed) if rng is None else rng

    def likelihood(self, config=None, value=None): ...

    def conditional_density(self, config=None): ...

    def sample(self, config=None, size=1): ...


class UniformKernelSampler(KernelSampler):
    def __init__(self, region: Region, rng=None, seed=None):
        super().__init__(rng, seed)
        self.region = region
        self.density = lambda x: float(x in self.region) / self.region.size()

    def likelihood(self, config=None, value=None):
        return int(self.conditional_density(config)(value))

    def conditional_density(self, config=None):
        return self.density


class RectangleKernelSampler(UniformKernelSampler):
    def __init__(self, x=0, y=0, width=1, height=1, rng=None, seed=None):
        region = Rectangle(x, y, width, height)
        super().__init__(region, rng, seed)

    def sample(self, config=None, size=1):
        xs = self.rng.uniform(
            self.region.bottomleft_x, self.region.topright_x, size=size
        )
        ys = self.rng.uniform(
            self.region.bottomleft_y, self.region.topright_y, size=size
        )
        return np.array((xs, ys)).transpose()


class IndexDiscreteSampler(KernelSampler):
    def __init__(self, density=None, rng=None, seed=None):
        super().__init__(rng, seed)
        self.density = StaticOrDynamic(density)
        # self.n = StaticOrDynamic(n)

    def get_density(self, config=None):
        return self.density(config)

    @classmethod
    def uniform_over(cls, n, rng=None, seed=None):
        density = np.ones(n) / n
        return cls(density=density, rng=rng, seed=seed)

    def sample(self, config=None, size=1):
        density = self.get_density(config=config)
        onehots = self.rng.multinomial(1, size=size, pvals=density)
        return onehots.argmax(axis=1)

    def likelihood(self, config=None, value=None):
        if isinstance(value, int):
            density = self.get_density(config)
            return density[value] if 0 <= value < len(density) else 0
        return 0

    def conditional_density(self, config=None):
        return self.get_density(config)


class StatesKernelSampler(KernelSampler):
    def __init__(self, states, density, rng=None, seed=None):
        super().__init__(rng, seed)
        self.states = StaticOrDynamic(
            states if callable(states) else np.asarray(states)
        )
        self.index_sampler = IndexDiscreteSampler(density, rng=self.rng)

    @classmethod
    def uniform_over(cls, states, rng=None, seed=None):
        density = np.ones(len(states)) / len(states)
        return cls(states=states, density=density, rng=rng, seed=seed)

    def get_states(self, config=None):
        return self.states(config)

    def sample_idx(self, config=None, size=1):
        return self.index_sampler.sample(config=config, size=size)

    def likelihood_idx(self, config=None, value=0):
        return self.index_sampler.likelihood(config=config, value=value)

    def likelihood(self, config=None, value=None):
        states = self.get_states(config)
        try:
            idx = next(i for i, val in enumerate(states) if val == value)
        except StopIteration:
            return 0
        return self.likelihood_idx(config=config, value=idx)

    def sample(self, config=None, size=1):
        states = np.asarray(self.get_states(config))
        density = self.index_sampler.get_density(config)
        if len(states) > len(density):
            warnings.warn("Extra states beyond density length will be ignored.")
        if len(states) < len(density):
            raise ValueError("Density refers to more states than provided.")
        # print(states, density)
        idx = self.sample_idx(config=config, size=size)
        return states[idx]

    def conditional_density(self, config=None):
        return self.index_sampler.conditional_density(config=config)


class CompositeKernelSampler(KernelSampler):
    def __init__(self, kernels, probs, rng=None, seed=None):
        super().__init__(rng, seed)
        self.kernels = StaticOrDynamic(kernels)
        self.kernels_sampler = IndexDiscreteSampler(density=probs, rng=self.rng)

    def likelihood(self, config=None, kernel_idx=None, value=None):
        return self.conditional_density(config=config, kernel_idx=kernel_idx)(value)

    def conditional_density(self, config=None, kernel_idx=None):
        kernels = self.kernels(config)

        return kernels[kernel_idx].conditional_density(config=config)

    def sample(self, config=None, size=1):
        kernels = self.kernels(config)
        idx = self.kernels_sampler.sample(config=config, size=size)

        samples = [
            (i, kernels[i].sample(config=config, size=1)[0]) for i in np.atleast_1d(idx)
        ]
        return samples


if __name__ == "__main__":
    # dcs = StatesKernelSampler(
    #     states=[1, 2, 5, 4], density=lambda config: config / np.sum(config)
    # )
    dcs = StatesKernelSampler.uniform_over(states=[0, 1, 2])
    # dcs = IndexDiscreteSampler.uniform_over(n=3)

    # print(dcs.sample(config=[1, 2, 3], size=10))
    # print(dcs.likelihood_idx(config=[1, 2, 3, 4], value=1))
    # print(dcs.conditional_density(config=[1, 2, 35]))

    from cProfile import Profile
    from pstats import SortKey, Stats

    with Profile() as profile:
        for i in range(20000):
            dcs.sample(config=[1, 2, 3], size=1)
        #  dcs.sample(config=[1, 2, 3], size=20000)
        (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats())
