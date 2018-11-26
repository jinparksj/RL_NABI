run.py: Training File to generate PKL file, which has leaning optimal policy
EXAMPLE of Arguments
''--env=Nabi-v0 --num_timesteps=10000000 --save_path /home/jin/project/rlnabi/PPO/data/params_20181026_10M.pkl'

sim_policy (On development): Simulation file using PKL file generating by run.py. It shows the simulation proving that NABI walks properly


ISSUE: (On Going)
Got MuJoCo Warning: Nan, Inf or huge value in QVEL at DOF 0
raise MujocoException('Got MuJoCo Warning: {}'.format(warn))
mujoco_py.builder.MujocoException: Got MuJoCo Warning: Nan, Inf or huge value in QVEL at DOF 0. The simulation is unstable. Time = 0.2600.

