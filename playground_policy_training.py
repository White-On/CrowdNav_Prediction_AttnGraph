import gym.wrappers
from rl.networks.envs import make_env
# from rl.networks.envs import make_vec_envs
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from rl.networks.storage import RolloutStorage
from crowd_sim.envs import *
from rl.networks.model import Policy


from rich import print
import torch
from rl import ppo
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from logger import logging_setup
import pandas as pd
import numpy as np
import gym



def main():
    from importlib import import_module
    model_dir = 'trained_models/env_experiment'

    log_path = Path(model_dir) / 'playground_env.log'
    logging_setup(log_path)


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
    
    arena_size = env_config.sim.arena_size
    arena_viz_factor = 2

    # set up visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-arena_size*arena_viz_factor, arena_size*arena_viz_factor)
    ax.set_ylim(-arena_size*arena_viz_factor, arena_size*arena_viz_factor)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.ion()
    plt.show()

    seed = np.random.randint(0, 1000)
    nb_enviroments = 5

    # env = make_env("CrowdSimCar-v0", seed, 1, "_", True,config=env_config, ax=ax)
    env = gym.vector.SyncVectorEnv(
        [make_env("CrowdSimCar-v0", seed, i, "_", True,config=env_config, ax=ax, envNum=nb_enviroments) for i in range(nb_enviroments)]
    )
    

    num_steps = 200
    num_episodes = 1
    log_file = 'env_experiment.log'
    save = True
    log_results_episodes = {'episode':[], 'status':[], 'reward':[], 'steps':[]}

    speed_action_space = [env.envs[0].action_space.low[0], env.envs[0].action_space.high[0]]
    delta_action_space = [env.envs[0].action_space.low[1], env.envs[0].action_space.high[1]]
    

    for episode in range(num_episodes):
        env.reset()
        env.envs[0].robot.full_reset()
        for step in range(num_steps):
            env.envs[0].render()

            angle_from_goal = env.envs[0].robot.get_angle_from_goal()
            angle_to_take = np.clip(angle_from_goal, delta_action_space[0], delta_action_space[1])

            distance_from_humans = env.envs[0].compute_distance_from_human()
            closest_human_distance = np.min(distance_from_humans)

            speed = np.clip(closest_human_distance*0.2, 0, 1)
            obs, reward, done, info = env.step([speed, angle_to_take])
            
            # obs, reward, done, info = env.step(action)

            
            # print(f"{obs = }")
            logging.info(f'Step: {step+1}, reward value: {reward[0]:.2f}, done: {done[0]}, status: {info[0].get("info")}')
            if done:
                logging.info(f'Episode {episode+1} finished at step {step+1}, status: {info[0].get("info")}')
                if save:
                    log_results_episodes['episode'].append(episode)
                    log_results_episodes['status'].append(info[0].get('info').__class__.__name__)
                    log_results_episodes['reward'].append(reward[0])
                    log_results_episodes['steps'].append(step+1)
                break
            

    plt.show()
    if save:
        pd.DataFrame(log_results_episodes).to_csv(log_file, index=False)

if __name__ == '__main__':
    main()