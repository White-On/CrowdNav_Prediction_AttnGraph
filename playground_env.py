from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
from gym_file.envs.crowd_sim_car_complex_obs import CrowdSimCarComplexObs
from logger import logging_setup

import logging
import pandas as pd
import numpy as np
import gym
import gin


@gin.configurable
def create_env(
    episode_time: int,
    nb_pedestrians: int = 10,
    robot_is_visible: bool = True,
    nb_goals_agent: int = 5,
    scenario: str = None,
    context_max_size: int = 10,
    ghost_mode: bool = False,
):
    env = gym.make(
        "CrowdSimCar-v2",
        render_mode="human",
        episode_time=episode_time,
        nb_pedestrians=nb_pedestrians,
        disable_env_checker=True,
        load_scenario=scenario,
        robot_is_visible=robot_is_visible,
        nb_goals_agent=nb_goals_agent,
        context_max_size=context_max_size,
        ghost_mode=ghost_mode,
        title="playground",
    )
    return env


def main():
    gym.logger.set_level(40)
    log_file = "env_experiment.log"
    logging_setup(log_file, level=logging.INFO)
    gin.parse_config_file("config.gin")
    # np.random.seed(0)

    num_steps = 400
    random_behavior = True
    num_episodes = 1
    # logging.info(gym.envs.registry.keys())

    # env = CrowdSimCar(render_mode='human', episode_time=num_steps, nb_pedestrians=20)
    env = create_env(episode_time=num_steps)
    # logging.info(f'{env.observation_space.shape[0]}')
    save = False
    log_results_episodes = {"episode": [], "status": [], "reward": [], "steps": []}

    # sample different position points on the map

    for episode in range(num_episodes):
        env.reset()
        env.render()
        total_reward = 0

        for step in range(num_steps):
            action = env.robot.predict_what_to_do()
            if random_behavior:
                random_angle = np.random.uniform(-np.pi / 6, np.pi / 6)
                random_acceleration = np.random.uniform(-0.2, 0.5)
                # random_acceleration = (
                #     -0.2 if env.robot.velocity_norm > env.robot.desired_speed else 0.2
                # )
                random_acceleration = (
                    env.robot.desired_speed - env.robot.velocity_norm
                ) / env.robot.delta_t
                # logging.info(
                #     f"{env.robot.velocity_norm = }, {random_acceleration = }"
                # )
                action[0] = random_acceleration
                # action[1] = random_angle
                # action[1] = -np.pi / 6

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if isinstance(obs, list):
                obs = np.array(obs)

            # logging.info(f"{obs = }")
            # logging.info(f"{obs['robot_node'][2:] = }")
            # logging.info(
            #     f"Step: {step+1}, reward: {reward:.2f}, done: {done}, status: {info['info']}"
            # )
            if done:
                logging.info(
                    f"Episode {episode+1} finished at step {step+1}, status: {info['info']}"
                )
                if save:
                    log_results_episodes["episode"].append(episode)
                    log_results_episodes["status"].append(info["info"])
                    log_results_episodes["reward"].append(reward)
                    log_results_episodes["steps"].append(step + 1)
                # break

        logging.info(f"Total reward: {total_reward:.2f}")
    env.close()
    if save:
        pd.DataFrame(log_results_episodes).to_csv(log_file, index=False)


if __name__ == "__main__":
    main()
