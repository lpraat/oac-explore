import os

from gym import Env
from gym.spaces import Box, Discrete, Tuple
import numpy as np


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


def domain_to_env(name):

    # from gym.envs.mujoco import HalfCheetahEnv, \
    #     InvertedPendulumEnv, HumanoidEnv, \
    #     HopperEnv, AntEnv, Walker2dEnv

    from envs.ant import Ant
    from envs.ant_mod import Ant as AntMod
    from envs.grid_world import GridWorldContinuous
    from envs.humanoid import Humanoid

    return {
        # 'invertedpendulum': InvertedPendulumEnv,
        # 'humanoid': HumanoidEnv,
        # 'halfcheetah': HalfCheetahEnv,
        # 'hopper': HopperEnv,
        # 'ant': AntEnv,
        # 'walker2d': Walker2dEnv

        'AntEscape': AntMod,
        'AntJump': Ant,
        'AntNavigate': Ant,
        'HumanoidUp': Humanoid,

        'GridGoal1': GridWorldContinuous,
        'GridGoal2': GridWorldContinuous,
        'GridGoal3': GridWorldContinuous,
    }[name]


def domain_to_epoch(name):
    raise NotImplementedError("Use command line arg instead.")
    # return {
    #     # 'invertedpendulum': 300,
    #     # 'humanoid': 9000,
    #     # 'halfcheetah': 5000,
    #     # 'hopper': 2000,
    #     # 'ant': 5000,
    #     # 'walker2d': 5000,

    #     'AntEscape': 500,
    #     'AntJump': 1000,
    #     'AntNavigate': 1000,
    #     'HumanoidUp': 2000,

    #     'GridGoal1': 100,
    #     'GridGoal2': 100,
    #     'GridGoal3': 100,


    # }[name]


def env_producer(domain, seed):

    sparse_reward_lambda = {
        'GridGoal1': gridgoal1_sparse_reward,
        'GridGoal2': gridgoal2_sparse_reward,
        'GridGoal3': gridgoal3_sparse_reward,
        'AntEscape': antescape_sparse_reward,
        'AntJump': antjump_sparse_reward,
        'AntNavigate': antnavigate_sparse_reward,
        'HumanoidUp': humanoidup_sparse_reward
    }[domain]

    env = domain_to_env(domain)(get_reward=sparse_reward_lambda)
    env.seed(seed)

    # env = NormalizedBoxEnv(env)

    return env


def gridgoal1_sparse_reward(s, r, d, i):
    if np.linalg.norm(s - np.array([5, 5], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def gridgoal2_sparse_reward(s, r, d, i):
    if np.linalg.norm(s - np.array([2, 5], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def gridgoal3_sparse_reward(s, r, d, i):
    if np.linalg.norm(s - np.array([5, 2], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def antescape_sparse_reward(s, r, d, i):
    _self = i['self']
    l1 = _self.unwrapped.get_body_com('aux_1')[2]
    l2 = _self.unwrapped.get_body_com('aux_2')[2]
    l3 = _self.unwrapped.get_body_com('aux_3')[2]
    l4 = _self.unwrapped.get_body_com('aux_4')[2]
    thresh = 0.8
    if l1 >= thresh and l2 >= thresh and l3 >= thresh and l4 >= thresh:
        return 1, True
    else:
        return 0, False

def antnavigate_sparse_reward(s, r, d, i):
    if s[0] >= 7:
        return 1, True
    else:
        return 0, False

def antjump_sparse_reward(s, r, d, i):
    if s[2] >= 3:
        return 1, True
    else:
        return 0, False

def humanoidup_sparse_reward(s, r, d, i):
    if s[2] >= 1:
        return 1, True
    else:
        return 0, False