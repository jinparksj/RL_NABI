from gym.envs.registration import register

def initNabi(policy = None):
    register(
        id='Nabi-v0',
        entry_point='PPO.envs.mujoco:NabiEnv',#'''VERY IMPORTANT!!!!!''' for IMPORT
        max_episode_steps=2000,
        reward_threshold=6000.0,
        nondeterministic=True,
        kwargs={'policy': policy}

            # 'observation_dim': observation_dim, # 22
            #      'policy_output_dim': policy_output_dim, # 4??
            #      'legs': legs} #????
    )
    return None

"""
def register(self, id, **kwargs):
        if id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)

Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        trials (int): The number of trials to average reward over
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        local_only: True iff the environment is to be used only on the local machine (e.g. debugging envs)
        kwargs (dict): The kwargs to pass to the environment class
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags

"""
