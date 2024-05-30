from rl.networks.envs import make_env
from rl.networks.envs import make_vec_envs
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from crowd_sim.envs import *


from rich import print
import matplotlib.pyplot as plt
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

    # set up visualization
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-6.5, 6.5) # 6
    ax.set_ylim(-6.5, 6.5)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # ax.set_xlabel('x(m)', fontsize=16)
    # ax.set_ylabel('y(m)', fontsize=16)
    plt.ion()
    plt.show()

    seed = np.random.randint(0, 1000)

    env = make_env("CrowdSimCar-v0", seed, 1, "_", True,config=env_config, ax=ax)
    env = DummyVecEnv([env])
    print(f"{env.observation_space=}")
    # print(env.action_space.low)
    # print(env.action_space.high)
    
    env.reset()
    # print(env.envs[0].generate_ob(True))
    reward_value, done, status = env.envs[0].calc_reward()
    print(f'reward_value: {reward_value:.2f}, done: {done}, status: {status}')
	
    env.envs[0].render()

    plt.show()

if __name__ == '__main__':
    main()