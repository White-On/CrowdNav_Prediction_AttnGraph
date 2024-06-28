import gymnasium as gym
import numpy as np


from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import chime
import logging
from logger import logging_setup

def main():
    logging_setup("DDPG_evaluation.log", level=logging.INFO)

    logging.info(gym.envs.registry.keys())
    env = gym.make(
            "CrowdSimCar-v1",
            render_mode="human",
            episode_time=200,
            nb_pedestrians=10,
            disable_env_checker=True,
            load_scenario="front",
        )

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("ddpg_CrowdSimCar")
    vec_env = model.get_env()

    del model # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_pendulum")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        env.render()

if __name__ == "__main__":
    main()