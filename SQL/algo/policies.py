import tensorflow as tf
import numpy as np
from util.serializable import Serializable

load_params = True

'''
============================NNQ <- MLP============================
'''

class Parameterized(object):
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}


    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params()))
            self.set_param_values(d["params"])

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError


    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]


    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]



class MLPFunction(Parameterized, Serializable):
    def __init__(self, inputs, name, hidden_layer_sizes):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._inputs = inputs
        self._layer_sizes = list(hidden_layer_sizes) + [1]

        self._output = self._output_for(self._inputs)

    def _output_for(self, inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            out = feedforward_net(
                inputs=inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes)

        return out[..., 0]

    def _eval(self, inputs):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf.get_default_session().run(self._output, feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope += '/' + self._name if scope else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


"""Neural Network Policy

Policies: object
    Neural Network Policy - class template
    Stochastic Neural Network Policy - implementation of SNN policy
"""

def feedforward_net(inputs,
                    layer_sizes,
                    activation_fn=tf.nn.relu,
                    output_nonlinearity=None):

    def bias(n_units):
        return tf.get_variable(
            name='bias', shape=n_units, initializer=tf.zeros_initializer())

    def linear(x, n_units, postfix=None):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        # `tf.tensordot` supports broadcasting
        return tf.tensordot(x, weight, axes=((-1, ), (0, )))

    out = 0
    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('layer_{i}'.format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    out += linear(input_tensor, layer_size, j)
            else:
                out = linear(out, layer_size)

            out += bias(layer_size)

            if i < len(layer_sizes) - 1 and activation_fn:
                out = activation_fn(out)

    if output_nonlinearity:
        out = output_nonlinearity(out)

    return out



def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]



'''
============================StochasticNNPolicy============================
'''


class Policy(Parameterized):
    def __init__(self, env_spec):
        Parameterized.__init__(self)
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError

    def reset(self, dones=None):
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        Return keys and shapes for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass

class NNPolicy(Policy, Serializable):
    def __init__(self, env_spec, obs_pl, action, scope_name=None):
        Serializable.quick_init(self, locals())

        self._obs_pl = obs_pl
        self._action = action
        self._scope_name = (tf.get_variable_scope().name
                            if not scope_name else scope_name)
        super(NNPolicy, self).__init__(env_spec)



class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec,
                 hidden_layer_sizes,
                 squash=True,
                 name='policy',
                 ac_dim=np.int64(3)):
        Serializable.quick_init(self, locals())

        self._action_dim = ac_dim
        self._observation_dim = env_spec.observation_flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation')

        self._actions = self.actions_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def actions_for(self, observations, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        return tf.tanh(raw_actions) if self._squash else raw_actions





