from sb3_contrib import RecurrentPPO
import gymnasium as gym
import numpy as np

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup


def main():
    logging_setup("Recurrent_PPO_evaluation.log", level=logging.DEBUG)
    episode_time = 200
    eval = False
    total_timesteps = 2_000_000
    save_every_n_timesteps = 10_000
    nb_learnging_cycles = total_timesteps // save_every_n_timesteps
    model_file = "recurrent_ppo_CrowdSimCar"

    env = gym.make(
        "CrowdSimCar-v0",
        render_mode="human",
        episode_time=episode_time,
        nb_pedestrians=10,
        disable_env_checker=True,
        load_scenario="random",
        robot_is_visible=True,
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
    )

    if eval:
        env = model.get_env()
        model = RecurrentPPO.load(model_file)
        obs = env.reset()
        # cell and hidden state of the LSTM
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        for _ in range(episode_time):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
            obs, rewards, dones, info = env.step(action)
            episode_starts = dones
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
    model = RecurrentPPO.load(model_file)

    obs = vec_env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)

    for _ in range(episode_time):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        env.render()


if __name__ == "__main__":
    main()
