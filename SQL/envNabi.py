from gym.envs.registration import register
import numpy as np
from gym import utils
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os

def initNavi(leg_indices, obs_dim = 22, act_dim = 2):
    '''
    obs_dimension: XYZ
    act_dimenstion: total 4 dof, 2 dof at each leg
    NEED TO DEFINE AGAIN

    TWO LEGS and 4DOF
    '''

    leg = len(leg_indices)

    register(id = 'Navi-v0',
             entry_point='NAVIEnv', #NEED TO MAKE!!!! -> Import below class Navi, I THINK IT MAY INCUR ERROR
             max_episode_steps = 1000,
             reward_threshold=6000.0,
             nondeterministic=True,
             kwargs={
                 'observation_dim': obs_dim,
                 'policy_output_dim': act_dim,
                 'legs': leg,
                 'leg_indices': leg_indices,
             })
    return None

"""
NAVI Model

    Diagram
    -------
               -------
        (Rear) |     | (Front)
          [1]  -------   [0]
             (0)/  \ (2)
            (1)/    \(3)
              /      \  at ground



    Action Space
    ------------

        action = np.array([0.0, 0.0, # Front leg [0]
                            0.0, 0.0]) # Rear leg [1]

    Position limits
    ---------------




"""

class NAVIEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.MAPPING = {0: [0, 1],
                        1: [2, 3]}

        self.REST_POSE = np.array([0.0, 0.0, 0.0, 0.0]) #NEED TO DEFINE WITH ALEXIE

        self.LEGS_UP = np.array([0.0, 0.0, 0.0, 0.0]) #NEED TO DEFINE WITH ALEXIE
        self.COXA_LIMIT = (-1/3, 1/3)
        self.KNEE_LIMIT = (-1/2, 1/2)
        self.COXA_INDEX = [0, 2]
        self.KNEE_INDEX = [1, 3]

        self.legs = kwargs['legs']
        self.act_dim = self.legs * 2

        self.leg_indices = kwargs['leg_indices']
        self.act_index, self.other_index = self.get_index()

        self.obs_dim = kwargs['observation_dim']
        self.input_dim = kwargs['policy_output_dim']

        self.move = self.REST_POSE.copy()[0:2]
        self.pos = self.REST_POSE.copy()

        xml_path = os.path.dirname(os.path.abspath(__file__)) + '/envs/model/xmlfilename.xml' #NEED TO MODIFY XML FILE NAME

        MujocoEnv.__init__(self, xml_path, 5) #frame skip : 5, Initialize self

        utils.EzPickle.__init__(self)

    def get_index(self):
        indices = []
        mapping = self.MAPPING.copy()
        for idx in self.leg_indices:
            indices.append(mapping.pop(idx))
        other_index = np.array(list(mapping.values()))

        return np.array(indices).flatten(), other_index

    # def step(self, a):
    #
    #
    #
    # def feed_action(self):
    #     """
    #     Feed action to lower level policy
    #     :return:
    #     """
    #