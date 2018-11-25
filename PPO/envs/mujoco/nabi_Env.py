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

        Action limits for all actuators: [-1 1]

        Right Hip: 0~90 degrees ==> 0~pi/2 ==>  1 ctr/(pi rad) * (pi/3 rad)  ==> 0~1/2
            [-1/3 1/3]
        Femur: +/-90 degrees ==> pi/2 ==>  1 ctr/(pi rad) * (pi/2 rad)  ==> 1/2
            [-1/2 1/2]
        Tibia: Same as Femur
            [-1/2 1/2]


"""

class NabiEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):

        self.MAPPING = {0: [0, 1],
                        1: [2, 3]}

        self.REST_POSE = np.array([1/4, -1/4, -1/8, 1/8])  # NEED TO DEFINE WITH ALEXIE
        #0: RIGHT HIP, 1: RIGHT KNEE, 2: LEFT HIP, 3: LEFT KNEE
        # self.LEGS_UP = np.array([0.0, 0.0, 0.0, 0.0])  # NEED TO DEFINE WITH ALEXIE
        self.RIGHT_HIP_LIMIT = (0, 1/2)
        self.LEFT_HIP_LIMIT = (-1/2, 0)
        self.RIGHT_KNEE_LIMIT = (-1/4, 1/4)
        self.LEFT_KNEE_LIMIT = (-1/4, 1/4)

        self.RIGHT_HIP_INDEX = 0
        self.RIGHT_KNEE_INDEX = 1
        self.LEFT_HIP_INDEX = 2
        self.LEFT_KNEE_INDEX = 3

        self.KNEE_INDEX = [1, 3]
        self.HIP_INDEX = [0, 2]

        self.move = self.REST_POSE.copy()[0:2]
        self.advance(True)
        self.pos = self.REST_POSE.copy()
        self.reward = 0

        self.policy = kwargs['policy']

        # xml_path = os.path.dirname(
        #     os.path.abspath(__file__)) + '/envs/model/past_NABI-v0.xml'  # NEED TO MODIFY XML FILE NAME
        xml_path = '/home/jin/project/rlnabi/PPO/envs/model/past_NABI-v0.xml'
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

    def feed_action(self, a):
        self.pos = a
        self.pos[self.RIGHT_HIP_INDEX] = np.clip(self.pos[self.RIGHT_HIP_INDEX], self.RIGHT_HIP_LIMIT[0], self.RIGHT_HIP_LIMIT[1])
        self.pos[self.LEFT_HIP_INDEX] = np.clip(self.pos[self.LEFT_HIP_INDEX], self.LEFT_HIP_LIMIT[0], self.LEFT_HIP_LIMIT[1])
        self.pos[self.RIGHT_KNEE_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_INDEX], self.RIGHT_KNEE_LIMIT[0], self.RIGHT_KNEE_LIMIT[1])
        self.pos[self.LEFT_KNEE_INDEX] = np.clip(self.pos[self.LEFT_KNEE_INDEX], self.LEFT_KNEE_LIMIT[0], self.LEFT_KNEE_LIMIT[1])
        return self.pos

    def advance(self, done):
        if done:
            self.pos = self.REST_POSE.copy()
        self.reward = 0.0

    def step(self, a):
        xposbefore = self.get_body_com("base_link")[0]
        yposbefore = self.get_body_com("base_link")[1]
        zposbefore = self.get_body_com("base_link")[2]

        a = self.feed_action(a)

        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("base_link")[0]
        yposafter = self.get_body_com("base_link")[1]
        zposafter = self.get_body_com("base_link")[2]

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
        self.advance(done)
        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl = -ctrl_cost,
            reward_contact = -contact_cost,
            reward_survive = survive_reward
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:], #self.sim.data.qpos.flat
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



