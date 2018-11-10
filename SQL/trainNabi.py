import tensorflow as tf
import numpy as np
import os

from .envNabi import initNavi
from .envs.gym_env import GymEnv
from .envs.normalize_env import normalize
from .util.replay_buffer import SimpleReplayBuffer
from .util.sampler import SimpleSampler
from .algo.value_functions import NNQFunction
from .algo.policies import StochasticNNPolicy
from .algo.sql import SQLAlgorithm
from .algo.sql_kernel import adaptive_isotropic_gaussian_kernel

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

"""
NEED TO WORK FOR ARGS SIMPLIFICATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


SHARED_PARAMS = {
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128, #flexible for layer size
    'batch_size': 128,
    'max_pool_size': 1E6, #1M
    'n_train_repeat': 1,
    'epoch_length': 1000, #episode length
    'kernel_particles': 16, #?????
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 500 #1000??, temporal-difference learning, learning at time step td
    #snapshot???
}

ENV_PARAMS = {
    'prefix': 'Navi-v0',
    'env_name': 'navi',
    'max_path_length': 1000,
    'n_epochs': 500,
    'reward_scale': 30,
    'legs': 1
}

def training(variant):
    initNavi(leg_indices=0, obs_dim=22, act_dim=2)

    env = normalize(GymEnv('Navi-v0', log_dir=PROJECT_PATH + "/data"))

    leg_mapping = env._wrapped_env.env.ac_indices #indices for motor??????????????????????????????
    ac_dim = np.int64(env._wrapped_env.env.input_dim)

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=SHARED_PARAMS['max_pool_size'], ac_dim=ac_dim)

    sampler = SimpleSampler(
        max_path_length=ENV_PARAMS['max_path_length'],
        min_pool_size=ENV_PARAMS['max_path_length'],
        batch_size=SHARED_PARAMS['batch_size'],
        leg_mapping=leg_mapping
    )

    M = SHARED_PARAMS['layer_size']

    scope_n = PROJECT_PATH + '/data/test'

    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), ac_dim=ac_dim, name=scope_n + '_q_func')
    policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=(M, M), ac_dim=ac_dim, name=scope_n + '_policy')

    algorithm = SQLAlgorithm(
        epoch_length=SHARED_PARAMS['epoch_length'],
        n_epochs=ENV_PARAMS['n_epochs'],
        n_train_repeat=SHARED_PARAMS['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=SHARED_PARAMS['kernel_particles'],
        kernel_update_ratio=SHARED_PARAMS['kernel_update_ratio'],
        value_n_particles=SHARED_PARAMS['value_n_particles'],
        td_target_update_interval=SHARED_PARAMS['td_target_update_interval'],
        qf_lr=SHARED_PARAMS['qf_lr'],
        policy_lr=SHARED_PARAMS['policy_lr'],
        discount=SHARED_PARAMS['discount'],
        reward_scale=ENV_PARAMS['reward_scale'],
        save_full_state=False,
        ac_dim=ac_dim)

    algorithm.train()


# def lanuch()


if __name__ == '__main__':
    # launch_experiments()
    print('pause')
