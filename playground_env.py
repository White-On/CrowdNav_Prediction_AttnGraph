from rl.networks.envs import make_env
from rl.networks.envs import make_vec_envs
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from crowd_sim.envs import *


from rich import print
import logging
import numpy as np


def main():
    from importlib import import_module
    model_dir = 'trained_models/env_experiment'
    model_dir_temp = model_dir
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]
    # import arguments.py from saved directory
    # if not found, import from the default directory
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.arguments'
        logging.info(f'Importing arguments from {model_dir_string}.py')
        model_arguments = import_module(model_dir_string)
        get_args = getattr(model_arguments, 'get_args')
    except:
        logging.error('Failed to get get_args function from ', model_dir, '/arguments.py\
                \nImporting arguments from default directory.')
        from arguments import get_args

    algo_args = get_args()

    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
        logging.info(f'Importing Config from {model_dir_string}.py')

    except:
        logging.error('Failed to get Config function from ', model_dir, '/configs/config.py')
        from crowd_nav.configs.config import Config
    env_config = config = Config()

    env = make_env("CrowdSimCar-v0", 42, 1, "_", True,config=env_config)
    env = DummyVecEnv([env])
    print(f"{env.observation_space=}")
    # print(env.action_space.low)
    # print(env.action_space.high)
    
    env.reset()
    env.envs[0].generate_ob(True)

if __name__ == '__main__':
    main()