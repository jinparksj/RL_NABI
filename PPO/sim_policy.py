import time
import tensorflow as tf
import gym
import numpy as np
import os
import joblib
import multiprocessing
from collections import defaultdict

from PPO.baselines import logger
from PPO.baselines.common.tf_util import get_session
from PPO.baselines.common.cmd_util import make_vec_env, common_arg_parser
from PPO.baselines.common.vec_env.vec_normalize import VecNormalize
from PPO.ppo2 import Model
from PPO.baselines.common.policies import build_policy


from PPO.envs.__init__ import initNabi
initNabi()

#args have num_env, alg, seed, env, reward_scale
NUM_ENV = 0
ALG = 'ppo2'
SEED = None
ENV = 'Nabi-v0'
REWARD_SCALE = 1.0
NETWORK = 'mlp' #mlp, cnn, lstm, cnn_lstm, conv_only
NSTEPS = 1000
NMINIBATCHES = 4
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

###################################model.load(load_path)######################

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

# ACTION: RH, RK, LH, LK -> - + + -
def simulate():

    logger.log("Running trained model")
    env = build_env()
    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs
    nbatch = nenvs * NSTEPS
    nbatch_train = nbatch // NMINIBATCHES

    policy = build_policy(env, NETWORK) #, **network_kwargs)
    '''
    (policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
    max_grad_norm=max_grad_norm)
    '''
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, \
                  nsteps=NSTEPS, ent_coef=ENT_COEF, vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM)
    load_path = 'data/params_20181026_1M.pkl'
    model.load(load_path)

    obs = env.reset()
    def initialize_placeholders(nlstm=128,**kwargs):
        return np.zeros((NUM_ENV or 1, 2*nlstm)), np.zeros((1))
    state, dones = initialize_placeholders() #**extra_args
    while True:
        actions, _, state, _ = model.step(obs,S=state, M=dones)
        # actions=[[-0.5, 0.5, 0.5, -0.5]]
        obs, _, done, _ = env.step(actions)
        env.render()
        # print(obs[0, 7], obs[0, 8], obs[0, 9], obs[0, 10])
        # print('actions: ', actions)
        # print('obs: ', obs)
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            obs = env.reset()

    env.close()

def build_env():
    seed = SEED

    env_type, env_id = get_env_type(ENV)
    #env_type: 'mujoco', env_id:'Nabi-v0'


    config = tf.ConfigProto(allow_soft_placement=True,
                           intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)

    config.gpu_options.allow_growth = True
    get_session(config=config)

    env = make_vec_env(env_id, env_type, NUM_ENV or 1, seed, reward_scale=REWARD_SCALE)
    env = VecNormalize(env)

    return env

def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


#args have num_env, alg, seed, env, reward_scale

if __name__ == '__main__':
    simulate()
