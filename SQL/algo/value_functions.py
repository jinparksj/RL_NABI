import numpy as np
import tensorflow as tf

from .policies import MLPFunction
from util.serializable import Serializable


class NNQFunction(MLPFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100, 100),
                 name='q_function',
                 ac_dim=np.int64(3)):
        Serializable.quick_init(self, locals())

        self._Da = ac_dim
        self._Do = env_spec.observation_flat_dim

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._Da], name='actions')

        super(NNQFunction, self).__init__(
            inputs=(self._observations_ph, self._actions_ph),
            name=name,
            hidden_layer_sizes=hidden_layer_sizes)

    def output_for(self, observations, actions, reuse=False):
        return super(NNQFunction, self)._output_for(
            (observations, actions), reuse=reuse)

    def eval(self, observations, actions):
        return super(NNQFunction, self)._eval((observations, actions))
