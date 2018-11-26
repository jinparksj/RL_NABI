import os
from gym.envs.mujoco.mujoco_env import MujocoEnv
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py
from mujoco_py import MjSimState
from collections import namedtuple

xml_path = os.path.dirname(os.path.abspath(__file__)) + '/model/Nabi-v0.xml'  # NEED TO MODIFY XML FILE NAME

mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

viewer = MjViewer(sim)
modder = TextureModder(sim)

t = 0
# a = [[100, 0, 0, 0, 0]]
# qpos = [0., 0., 0.8459, 1., 0., 0., 0., \
#          0.4, -0.2, -0.4, 0.2]
#
# qvel = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
actions = [[0, 0, 0, 0]]
# state = MjSimState(qpos = qpos, time = 0.0, qvel = qvel, act = None, udd_state={})
# sim.reset()

while True:
    for name in sim.model.geom_names:
        modder.rand_all(name)

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
