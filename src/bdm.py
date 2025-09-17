import numpy as np
from src.kernels import (
    StatesKernelSampler,
    RectangleKernelSampler,
    IndexDiscreteSampler,
    CompositeKernelSampler,
)
from src.kernels import uniform_probs

from enum import Enum


class BDM_states(Enum):

    BIRTH = 0
    DEATH = 1
    MIGRATION = 2


class BirthDeathMigration:
    STATES = [state for state in BDM_states]

    def __init__(
        self,
        density,
        new_point_sampler=None,
        point_to_remove_sampler=None,
        migration_sampler=None,
        state_sampler_probs=None,
        rng=None,
        seed=None,
    ):
        self.rng = rng if rng else np.random.default_rng(seed)
        if new_point_sampler is None:
            new_point_sampler = RectangleKernelSampler(rng=self.rng)
        if point_to_remove_sampler is None:
            point_to_remove_sampler = IndexDiscreteSampler(
                density=uniform_probs, rng=self.rng
            )

        if migration_sampler is None:
            kernels = lambda config: [RectangleKernelSampler(rng=self.rng)] * len(
                config
            )
            migration_sampler = CompositeKernelSampler(
                kernels=kernels, probs=uniform_probs, rng=self.rng
            )

        self.density = density
        self.new_point_sampler = new_point_sampler
        self.point_to_remove_sampler = point_to_remove_sampler
        self.state_sampler = (
            StatesKernelSampler.uniform_over(states=self.STATES, rng=self.rng)
            if state_sampler_probs is None
            else StatesKernelSampler(
                states=self.STATES, density=state_sampler_probs, rng=self.rng
            )
        )
        self.migration_sampler = migration_sampler
        self.config = np.empty(shape=(0, 2))
        self.bdm = {
            BDM_states.BIRTH: self.birth_step,
            BDM_states.DEATH: self.death_step,
            BDM_states.MIGRATION: self.migration_step,
        }

    def run(self, num_iter, warm_up=0, callbacks=None):
        if callbacks is None:
            callbacks = []
        states = self.state_sampler.sample(
            size=warm_up + num_iter
        )  # wrong, if states not static.
        for i in range(warm_up):
            self.step(new_state=states[i])
        for i in range(warm_up, num_iter):
            args = self.step_with_reconstruction(new_state=states[i])
            for cb in callbacks:
                cb(*args)
            yield args[1] if args[4] else args[2]

    def _propose_core(self, config=None, new_state=None):
        if config is None:
            config = self.config
        if new_state is None:
            new_state = self.state_sampler.sample(config, size=1)[0]
        new_config, acceptance_probability, reconstruction_params = self.bdm[new_state](
            config
        )
        return new_state, new_config, acceptance_probability, reconstruction_params

    def propose(self, config=None):
        new_state, new_config, acceptance_probability, _ = self._propose_core(config)
        return new_state, new_config, acceptance_probability

    def propose_with_reconstruction(self, config=None):
        return self._propose_core(config)

    def _step_core(self, config=None, with_reconstruction=False, new_state=None):
        result = self._propose_core(config, new_state)
        new_state, new_config, acceptance_probability, reconstruction_params = result
        is_accepted = self.rng.uniform() <= acceptance_probability
        if config is None:
            config = self.config
            self.config = new_config if is_accepted else self.config

        if with_reconstruction:
            return (
                new_state,
                config,
                new_config,
                acceptance_probability,
                is_accepted,
                reconstruction_params,
            )
        else:
            return new_state, config, new_config, acceptance_probability, is_accepted

    def step(self, config=None, new_state=None):
        return self._step_core(config, with_reconstruction=False, new_state=new_state)

    def step_with_reconstruction(self, config=None, new_state=None):
        return self._step_core(config, with_reconstruction=True, new_state=new_state)

    def __h_birth(self, config, new_config):
        h = (
            self.density.parangelou(config, new_config[-1])
            * self.state_sampler.likelihood(new_config, BDM_states.DEATH)
            / self.state_sampler.likelihood(config, BDM_states.BIRTH)
            * self.point_to_remove_sampler.likelihood(new_config, len(new_config) - 1)
            / self.new_point_sampler.likelihood(config, new_config[-1])
        )
        return h

    def __h_death(self, config, new_config):
        return 1 / self.__h_birth(new_config, config)

    def birth_step(self, config=np.empty(shape=(0, 2))):
        new_point = self.new_point_sampler.sample(config)[0]

        new_config = np.vstack((config, new_point))

        acceptance_probability = min(1, self.__h_birth(config, new_config))
        return new_config, acceptance_probability, (new_point,)

    def death_step(self, config):
        if len(config) == 0:
            return config, 0, ()
        point_idx = self.point_to_remove_sampler.sample(config)[0].item()
        new_config = config.copy()
        new_config[point_idx] = new_config[-1]
        new_config = config[:-1]
        return new_config, min(1, self.__h_death(config, new_config)), (point_idx,)

    def migration_step(self, config):
        if len(config) == 0:
            return config, 0, ()
        point_idx, new_point = self.migration_sampler.sample(config=config, size=1)[0]
        #        config[[point_idx, -1]] = config[[-1, point_idx]] #swap
        new_config = config.copy()
        new_config[point_idx] = new_point

        h = (
            self.density.log_parangelou(config[:-1], new_point)
            - self.density.log_parangelou(config[:-1], config[-1])
            #        self.density.log_mixed_parangelou(config, point_idx, new_point)
            + np.log(
                self.migration_sampler.likelihood(
                    new_config, point_idx, config[point_idx]
                )
                / self.migration_sampler.likelihood(config, point_idx, new_point)
            )
        )
        acceptance_probability = np.exp(min(0, h))

        return new_config, acceptance_probability, (point_idx, new_point)

    @staticmethod
    def reconstruct_config(config, state, reconstruction_params):
        if state == BDM_states.BIRTH:
            new_config = np.vstack((config, reconstruction_params[0]))
        if state == BDM_states.MIGRATION:
            new_config = config.copy()
            new_config[reconstruction_params[0]] = reconstruction_params[1]
        if state == BDM_states.DEATH:
            new_config = np.delete(config, reconstruction_params[0], axis=0)
        return new_config


class HistoryTracker:
    def __init__(
        self,
        track_configs=True,
        track_states=True,
        track_acceptance=True,
        track_were_accepted=True,
    ):
        self.states = [] if track_states else None
        self.acc_probs = [] if track_acceptance else None
        self.were_accepted = [] if track_were_accepted else None
        self.configs = [] if track_configs else None

    def __call__(
        self,
        new_state,
        old_config,
        new_config,
        acceptance_probability,
        is_accepted,
        reconstruction_params=None,
    ):

        if self.states is not None:
            self.states.append(new_state)
        if self.acc_probs is not None:
            self.acc_probs.append(acceptance_probability)
        if self.were_accepted is not None:
            self.were_accepted.append(is_accepted)
        if self.configs is not None:
            self.configs.append(new_config if is_accepted else old_config)


class ConfigEvaluator:
    def __init__(self, function):
        self.func = function
        self.values = []

    def __call__(
        self,
        new_state,
        old_config,
        new_config,
        acceptance_probability,
        is_accepted,
        reconstruction_params=None,
    ):

        self.values.append(
            self.func(new_config) if is_accepted else self.func(old_config)
        )
