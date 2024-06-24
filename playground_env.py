from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
from logger import logging_setup

from rich import print
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import gym


def main():
    gym.logger.set_level(40)
    log_file = "env_experiment.log"
    logging_setup(log_file)

    num_steps = 500
    random_behavior = False
    num_episodes = 1

    # env = CrowdSimCar(render_mode='human', episode_time=num_steps, nb_pedestrians=20)
    env = gym.make(
        "CrowdSimCar-v0",
        render_mode="human",
        episode_time=num_steps,
        nb_pedestrians=20,
        disable_env_checker=True,
        robot_is_visible=True,
    )
    # logging.info(f'{env.observation_space.shape[0]}')
    save = False
    log_results_episodes = {"episode": [], "status": [], "reward": [], "steps": []}

    # sample different position points on the map

    for episode in range(num_episodes):
        env.reset()
        env.render()

        for step in range(num_steps):
            action = env.robot.predict_what_to_do()
            if random_behavior:
                random_angle = np.random.uniform(-np.pi / 6, np.pi / 6)
                action = [1, random_angle]
            obs, reward, done, info = env.step(action)
            env.render()

            if isinstance(obs, list):
                obs = np.array(obs)

            # logging.info(f"{obs.shape = }")
            logging.info(
                f"Step: {step+1}, reward: {reward:.2f}, done: {done}, status: {info}"
            )
            if done:
                logging.info(
                    f"Episode {episode+1} finished at step {step+1}, status: {info}"
                )
                if save:
                    log_results_episodes["episode"].append(episode)
                    log_results_episodes["status"].append(info.__class__.__name__)
                    log_results_episodes["reward"].append(reward)
                    log_results_episodes["steps"].append(step + 1)
                # break

    env.close()
    if save:
        pd.DataFrame(log_results_episodes).to_csv(log_file, index=False)


if __name__ == "__main__":
    main()
