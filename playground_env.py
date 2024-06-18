from crowd_sim_car import CrowdSimCar
from logger import logging_setup

from rich import print
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import gym


def main():
    gym.logger.set_level(40)
    log_file = 'env_experiment.log'
    logging_setup(log_file)

    num_steps = 200
    num_episodes = 1

    env = CrowdSimCar(render_mode='human', episode_time=num_steps, nb_pedestrians=20)

    save = False
    log_results_episodes = {'episode':[], 'status':[], 'reward':[], 'steps':[]}

    for episode in range(num_episodes):
        env.reset()
        for step in range(num_steps):
            agent_visible = env.all_agent_group.filter(lambda x: x.is_visible)
            other_agent_state = (agent_visible
                                 .filter(lambda x: x.id != env.robot.id)
                                 .filter(env.robot.can_i_see)
                                 .apply(lambda x: x.coordinates + x.speed))
            action = env.robot.predict_what_to_do(*other_agent_state)
            obs, reward, done, info = env.step(action)

            
            # logging.info(f"{obs = }")
            logging.info(f'Step: {step+1}, reward: {reward:.2f}, done: {done}, status: {info}')
            if done:
                logging.info(f'Episode {episode+1} finished at step {step+1}, status: {info}')
                if save:
                    log_results_episodes['episode'].append(episode)
                    log_results_episodes['status'].append(info.__class__.__name__)
                    log_results_episodes['reward'].append(reward)
                    log_results_episodes['steps'].append(step+1)
                # break
                
    
    env.close()
    if save:
        pd.DataFrame(log_results_episodes).to_csv(log_file, index=False)

if __name__ == '__main__':
    main()