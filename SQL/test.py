import numpy as np
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv
import gym



PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

print(PROJECT_PATH + '/data/test')


def get_fn(args, variant, i=0):
    q = i
    i = '' if i == 0 else i
    legs = len(list(set([int(k) for k in str(args.leg_indices[0])])))
    if args.other_policy is None or args.other_policy == 'None':
        other_policy = 'None'
    elif '/' in args.other_policy:
        other_policy = 'Policy'
    else:
        other_policy = args.other_policy
    full_experiment_name = args.control_lv \
                           + '-' + str(legs) \
                           + '-' + args.alg \
                           + '-' + other_policy \
                           + '-' + str(i).zfill(2)
    while os.path.isdir(os.path.abspath(__file__ + '../../..')
                        + '/data/local/'
                        + variant['prefix']
                        + '/' + args.exp_name
                        + '/' + full_experiment_name):
        q += 1
        full_experiment_name = args.control_lv \
                               + '-' + str(legs) \
                               + '-' + args.alg \
                               + '-' + other_policy \
                               + '-' + str(q).zfill(2)
    return full_experiment_name