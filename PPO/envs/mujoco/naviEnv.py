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
        self.reward = 0

        xml_path = os.path.dirname(
            os.path.abspath(__file__)) + '/envs/model/NABI-v0.xml'  # NEED TO MODIFY XML FILE NAME

        MujocoEnv.__init__(self, xml_path, 5)  # frame skip : 5, Initialize self
        utils.EzPickle.__init__(self)


    # def feed_action(self, a):
    #     """
    #     Feed action to policy
    #
    #     takes in the actions from higher level controller
    #     and feeds them into the sing
    #     :param a:
    #     :return:
    #     """

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        zposbefore = self.get_body_com("torso")[2]

        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        zposafter = self.get_body_com("torso")[2]

        forward_reward = (xposafter - xposbefore) / self.dt

        ctrl_cost = 0.5 * 1e-2 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        self.reward = reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        obs = self._get_obs()
        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl = -ctrl_cost,
            reward_contact = -contact_cost,
            reward_survive = survive_reward
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

