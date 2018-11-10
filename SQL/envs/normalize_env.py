import numpy as np
from gym.spaces import Box


from util.serializable import Serializable
from .env_base import ProxyEnv, Step



class NormalizedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(np.prod(env.observation_space.low.shape))
        self._obs_var = np.ones(np.prod(env.observation_space.low.shape))
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.


normalize = NormalizedEnv