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

    length
    ------

    0.0558
    0.39152
    0.39857
    total : 0.84589

    0.05 + 0.05between right and left

"""

class NabiEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):

        self.MAPPING = {0: [0, 1],
                        1: [2, 3]}

        self.REST_POSE = np.array([-1/4, 1/8, 1/4, -1/8])  # NEED TO DEFINE WITH ALEXIE
        #0: RIGHT HIP, 1: RIGHT KNEE, 2: LEFT HIP, 3: LEFT KNEE
        # self.LEGS_UP = np.array([0.0, 0.0, 0.0, 0.0])  # NEED TO DEFINE WITH ALEXIE
        # self.RIGHT_HIP_LIMIT = (0, 1/2)
        # self.LEFT_HIP_LIMIT = (-1/2, 0)
        self.RIGHT_HIP_LIMIT = (-1/3, -1/6) # -
        self.LEFT_HIP_LIMIT = (1/6, 1/3) # +
        self.RIGHT_KNEE_LIMIT = (0, 1/4) # +
        self.LEFT_KNEE_LIMIT = (-1/4, 0) # -

        self.RIGHT_HIP_INDEX = 0
        self.RIGHT_KNEE_INDEX = 1
        self.LEFT_HIP_INDEX = 2
        self.LEFT_KNEE_INDEX = 3

        self.RIGHT_HIP_BL_INDEX = 4
        self.RIGHT_KNEE_BL_INDEX = 5
        self.LEFT_HIP_BL_INDEX = 6
        self.LEFT_KNEE_BL_INDEX = 7


        self.KNEE_INDEX = [1, 3]
        self.HIP_INDEX = [0, 2]
        self.len_Femur = 0.39152
        self.len_Tibia = 0.39857
        self.dist_btwn = 0.035

        self.pos = self.REST_POSE.copy()
        self.reward = 0
        self.numofmotor = 4

        self.policy = kwargs['policy']

        # xml_path = os.path.dirname(
        #     os.path.abspath(__file__)) + '/envs/model/past_NABI-v0.xml'  # NEED TO MODIFY XML FILE NAME
        xml_path = '/home/jin/project/rlnabi/PPO/envs/model/Nabi-v0.xml'
        # xml_path = '/home/jin/project/rlnabi/PPO/envs/model/Nabi-v1_jump.xml'
        frame_skip = 1
        MujocoEnv.__init__(self, xml_path, frame_skip)  # frame skip : 5, Initialize self
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
        for i in range(len(a)):
            if (a[i] == float('NaN')):
                a[i] = 0
            if a[i] == float('Inf'):
                if i == self.RIGHT_HIP_INDEX:
                    a[i] = self.RIGHT_HIP_LIMIT[1]
                if i == self.LEFT_HIP_INDEX:
                    a[i] = self.LEFT_HIP_LIMIT[1]
                if i == self.RIGHT_KNEE_INDEX:
                    a[i] = self.RIGHT_KNEE_LIMIT[1]
                if i == self.LEFT_KNEE_INDEX:
                    a[i] = self.LEFT_KNEE_LIMIT[1]
            if a[i] == -float('Inf'):
                if i == self.RIGHT_HIP_INDEX:
                    a[i] = self.RIGHT_HIP_LIMIT[0]
                if i == self.LEFT_HIP_INDEX:
                    a[i] = self.LEFT_HIP_LIMIT[0]
                if i == self.RIGHT_KNEE_INDEX:
                    a[i] = self.RIGHT_KNEE_LIMIT[0]
                if i == self.LEFT_KNEE_INDEX:
                    a[i] = self.LEFT_KNEE_LIMIT[0]

        # Add noise for

        self.pos = a
        #1.
        # self.pos[self.RIGHT_HIP_INDEX] = np.clip(self.pos[self.RIGHT_HIP_INDEX], self.RIGHT_HIP_LIMIT[0], self.RIGHT_HIP_LIMIT[1])
        # self.pos[self.LEFT_HIP_INDEX] = np.clip(self.pos[self.LEFT_HIP_INDEX], self.LEFT_HIP_LIMIT[0], self.LEFT_HIP_LIMIT[1])
        # angle_right_KNEE = np.clip((self.dist_btwn + self.len_Femur * np.sin(self.pos[self.RIGHT_HIP_INDEX] * np.pi)) / self.len_Tibia, \
        #                            -1, 1)
        # angle_left_KNEE = np.clip((self.dist_btwn + self.len_Femur * np.sin(self.pos[self.LEFT_HIP_INDEX] * np.pi)) / self.len_Tibia, \
        #                           -1, 1)
        #
        # min_limit_right_knee = np.arcsin(angle_right_KNEE)
        # max_limit_left_knee = np.arcsin(angle_left_KNEE)
        #
        # self.pos[self.RIGHT_KNEE_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_INDEX], -min_limit_right_knee, self.RIGHT_KNEE_LIMIT[1])
        # self.pos[self.LEFT_KNEE_INDEX] = np.clip(self.pos[self.LEFT_KNEE_INDEX], self.LEFT_KNEE_LIMIT[0], max_limit_left_knee)

        #2.
        # self.pos[self.RIGHT_HIP_INDEX] = np.clip(self.pos[self.RIGHT_HIP_INDEX], self.RIGHT_HIP_LIMIT[0],
        #                                          self.RIGHT_HIP_LIMIT[1])
        # self.pos[self.LEFT_HIP_INDEX] = np.clip(self.pos[self.LEFT_HIP_INDEX], self.LEFT_HIP_LIMIT[0],
        #                                         self.LEFT_HIP_LIMIT[1])
        # angle_right_KNEE = np.clip(
        #     (self.dist_btwn + self.len_Femur * np.sin(self.pos[self.RIGHT_HIP_INDEX] * np.pi)) / self.len_Tibia, \
        #     -1, 1)
        #
        # angle_left_KNEE = np.clip(
        #     (self.dist_btwn + self.len_Femur * np.sin(self.pos[self.LEFT_HIP_INDEX] * np.pi)) / self.len_Tibia, \
        #     -1, 1)
        #
        # max_limit_right_knee = np.arcsin(angle_right_KNEE)
        # min_limit_left_knee = np.arcsin(angle_left_KNEE)
        #
        # self.pos[self.RIGHT_KNEE_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_INDEX], self.RIGHT_KNEE_LIMIT[0],
        #                                           max_limit_right_knee)
        # self.pos[self.LEFT_KNEE_INDEX] = np.clip(self.pos[self.LEFT_KNEE_INDEX], -min_limit_left_knee, self.LEFT_KNEE_LIMIT[1])


        #3.
        # self.pos[self.RIGHT_HIP_INDEX] = np.clip(self.pos[self.RIGHT_HIP_INDEX], self.RIGHT_HIP_LIMIT[0],
        #                                          self.RIGHT_HIP_LIMIT[1])
        # self.pos[self.LEFT_HIP_INDEX] = np.clip(self.pos[self.LEFT_HIP_INDEX], self.LEFT_HIP_LIMIT[0],
        #                                         self.LEFT_HIP_LIMIT[1])
        #
        # self.pos[self.RIGHT_KNEE_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_INDEX], self.RIGHT_KNEE_LIMIT[0],
        #                                           self.RIGHT_KNEE_LIMIT[1])
        # self.pos[self.LEFT_KNEE_INDEX] = np.clip(self.pos[self.LEFT_KNEE_INDEX], self.LEFT_KNEE_LIMIT[0],
        #                                          self.LEFT_KNEE_LIMIT[1])

        #4.

        self.pos[self.RIGHT_HIP_INDEX] = np.clip(self.pos[self.RIGHT_HIP_INDEX], -1, 1)
        self.pos[self.LEFT_HIP_INDEX] = np.clip(self.pos[self.LEFT_HIP_INDEX], -1, 1)

        self.pos[self.RIGHT_KNEE_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_INDEX], -1, 1)
        self.pos[self.LEFT_KNEE_INDEX] = np.clip(self.pos[self.LEFT_KNEE_INDEX], -1, 1)

        self.pos[self.RIGHT_HIP_BL_INDEX] = np.clip(self.pos[self.RIGHT_HIP_BL_INDEX], -1, 1)
        self.pos[self.LEFT_HIP_BL_INDEX] = np.clip(self.pos[self.LEFT_HIP_BL_INDEX], -1, 1)

        self.pos[self.RIGHT_KNEE_BL_INDEX] = np.clip(self.pos[self.RIGHT_KNEE_BL_INDEX], -1, 1)
        self.pos[self.LEFT_KNEE_BL_INDEX] = np.clip(self.pos[self.LEFT_KNEE_BL_INDEX], -1, 1)

        return self.pos

    def advance(self):
        self.pos = self.REST_POSE.copy()
        # self.reward = 0.0

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
        not_y_reward = 10 * (yposafter - yposbefore) / self.dt
        not_x_reward = 10 * (xposafter - xposbefore) / self.dt
        jump_reward = (zposafter - zposbefore) / self.dt
        actuator_cost = abs(self.data.actuator_velocity[:self.LEFT_KNEE_INDEX+1]).sum()
        # ctrl_cost = 0.5 * 1e-4 * np.square(a).sum()

        # contact_cost = 0.5 * 1e-5 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # cfrc_ext: com-based external force on body         (nbody x 6)
        # 3D rot; 3D tran, External torques and forces
        weighting = 0.0008
        survive_reward = 5.
        # reward = forward_reward - ctrl_cost + survive_reward - 10 * not_y_reward - weighting * actuator_cost
        reward = forward_reward + survive_reward - not_y_reward - weighting * actuator_cost
        # reward = jump_reward - ctrl_cost + survive_reward - not_y_reward - not_x_reward
        # reward = 10*jump_reward - ctrl_cost + survive_reward - not_y_reward - not_x_reward
        self.reward = reward

        state = self.state_vector()

        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0 #NEED TO FIX IT
        done = not notdone
        obs = self._get_obs() # NEED to Change JOINT + BACKLASH
        self.advance()
        return obs, reward, done, dict(
            reward_forward=forward_reward,
            # reward_ctrl = -ctrl_cost,
            reward_actuator = -actuator_cost,
            # reward_contact = -contact_cost,
            reward_survive = survive_reward,
            reward_not_y_reward = -not_y_reward
        )

    def _get_obs(self):
        act_RH_pos = np.array(self.sim.data.qpos[7] + self.sim.data.qpos[11]).reshape(1, )
        act_RK_pos = np.array(self.sim.data.qpos[8] + self.sim.data.qpos[12]).reshape(1, )
        act_LH_pos = np.array(self.sim.data.qpos[9] + self.sim.data.qpos[13]).reshape(1, )
        act_LK_pos = np.array(self.sim.data.qpos[10] + self.sim.data.qpos[14]).reshape(1, )

        act_qpos = np.concatenate([self.sim.data.qpos.flat[:7], act_RH_pos, act_RK_pos, act_LH_pos, act_LK_pos])

        act_RH_vel = np.array(self.sim.data.qvel[6] + self.sim.data.qvel[10]).reshape(1, )
        act_RK_vel = np.array(self.sim.data.qvel[7] + self.sim.data.qvel[11]).reshape(1, )
        act_LH_vel = np.array(self.sim.data.qvel[8] + self.sim.data.qvel[12]).reshape(1, )
        act_LK_vel = np.array(self.sim.data.qvel[9] + self.sim.data.qvel[13]).reshape(1, )

        act_qvel = np.concatenate([self.sim.data.qvel.flat[:6], act_RH_vel, act_RK_vel, act_LH_vel, act_LK_vel])

        return np.concatenate([
            act_qpos, #[:12] 3 + 4+ 10 = 17: torso of x, y, z / quaternion of torso / joint: 4 + 4 + 2
            act_qvel #[:11] 3 + 3+ 10 = 16: torso vel of x, y, z / euler velocity of torso / joint: 4 + 4 + 2
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # qpos = self.init_qpos
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5



