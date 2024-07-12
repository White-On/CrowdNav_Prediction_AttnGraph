from stable_baselines3 import PPO
import gymnasium as gym

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup


def main():
    logging_setup("PPO_evaluation.log", level=logging.DEBUG)
    episode_time = 200
    eval = True
    total_timesteps = 2_000_000
    save_every_n_timesteps = 10_000
    nb_learnging_cycles = total_timesteps // save_every_n_timesteps
    model_file = "ppo_CrowdSimCar" 

    env = gym.make(
        "CrowdSimCar-v0",
        render_mode="human",
        episode_time=episode_time,
        nb_pedestrians=0,
        disable_env_checker=True,
        load_scenario=None,
        robot_is_visible=True,
        nb_goals_agent = 1,
    )

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
        learning_rate=linear_schedule(1e-3),
    )

    if eval:
        env = model.get_env()
        model = PPO.load(model_file)
        obs = env.reset()

        for _ in range(episode_time):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        return

    for i in range(nb_learnging_cycles):
        model.learn(
            total_timesteps=save_every_n_timesteps,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        model.save(model_file)

    vec_env = model.get_env()
    model = PPO.load(model_file)

    obs = vec_env.reset()

    for _ in range(episode_time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()


if __name__ == "__main__":
    main()
