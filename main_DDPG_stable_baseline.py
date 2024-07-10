import gymnasium as gym
import numpy as np


from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
)

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup


def main():
    logging_setup("DDPG_evaluation.log", level=logging.INFO)
    episode_time = 500
    eval = False
    total_timesteps = 5_000_000
    save_every_n_timesteps = 10_000
    nb_learnging_cycles = total_timesteps // save_every_n_timesteps
    model_file = "ddpg_CrowdSimCar"

    # logging.info(gym.envs.registry.keys())
    env = gym.make(
        "CrowdSimCar-v0",
        render_mode="human",
        episode_time=episode_time,
        nb_pedestrians=10,
        disable_env_checker=True,
        load_scenario=None,
    )

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="runs",
    )
    if eval:

        vec_env = model.get_env()
        model = DDPG.load(model_file)
        obs = vec_env.reset()

        for _ in range(episode_time):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            env.render()

        return

    for i in range(nb_learnging_cycles):
        model.learn(
            total_timesteps=save_every_n_timesteps,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        model.save(model_file)

    vec_env = model.get_env()
    model = DDPG.load(model_file)

    obs = vec_env.reset()

    for _ in range(episode_time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        env.render()


if __name__ == "__main__":
    main()
