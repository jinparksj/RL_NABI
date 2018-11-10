import numpy as np
from gym import utils
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os

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

class NabiEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        def __init__(self, **kwargs):
            self.MAPPING = {0: [0, 1],
                            1: [2, 3]}

            self.REST_POSE = np.array([0.0, 0.0, 0.0, 0.0])  # NEED TO DEFINE WITH ALEXIE

            self.LEGS_UP = np.array([0.0, 0.0, 0.0, 0.0])  # NEED TO DEFINE WITH ALEXIE
            self.COXA_LIMIT = (-1 / 3, 1 / 3)
            self.KNEE_LIMIT = (-1 / 2, 1 / 2)
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

            xml_path = os.path.dirname(
                os.path.abspath(__file__)) + '/envs/model/xmlfilename.xml'  # NEED TO MODIFY XML FILE NAME

            MujocoEnv.__init__(self, xml_path, 5)  # frame skip : 5, Initialize self

            utils.EzPickle.__init__(self)
