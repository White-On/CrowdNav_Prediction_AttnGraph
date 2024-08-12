from stable_baselines3 import PPO
from pathlib import Path
import gymnasium as gym
import argparse
import gin

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup


def parse_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    # mode eval ou pas
    parser.add_argument("-e", "--eval", action="store_true")
    # pour le logger le niveau
    parser.add_argument("-v", "--verbose", action="store_true")
    # nom fichier log tensorflow
    parser.add_argument("-l", "--log", type=str, default="PPO")
    # nom de fichier de configuration gin
    parser.add_argument("-c", "--config", type=str, default="config.gin")
    # nom du fichier du model sauvegardé
    parser.add_argument("-m", "--model", type=str, default="ppo_CrowdSimCar")
    return parser.parse_args()


def extract_specials_feature(config_file: str) -> str:
    # the "special features" are always seperated by a double empty line
    # we read the lines from the bottom to the top and stop when we find the first empty line
    # we return the special features as a string
    with open(config_file, "r") as f:
        lines = f.readlines()
        special_features = ""
        for line in reversed(lines):
            if line == "\n":
                break
            clean_line = line.split(".")[1]
            special_features = clean_line + special_features

    return special_features


@gin.configurable
def create_env(
    episode_time: int,
    nb_pedestrians: int = 10,
    robot_is_visible: bool = False,
    nb_goals_agent: int = 5,
    scenario: str = None,
    context_max_size: int = 10,
    ghost_mode: bool = False,
    title: str = None,
    learning_state: float = 1.0,
) -> gym.Env:
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
        title=title,
        learning_state=learning_state,
    )
    return env


@gin.configurable(denylist=["env"])
def init_model(
    env: gym.Env,
    gamma: float = 0.99,
) -> PPO:
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
        gamma=gamma,
        # learning_rate=linear_schedule(1e-3),
    )


@gin.configurable
def init_params(
    episode_time=4000,
    total_timesteps=2_000_000,
    save_every_n_timesteps=10_000,
) -> tuple:
    return episode_time, total_timesteps, save_every_n_timesteps


def main() -> None:
    args = parse_args()
    logging_setup(
        "PPO_evaluation.log", level=logging.DEBUG if args.verbose else logging.INFO
    )
    # check if the config file exists
    config_file_exist = Path(args.config).exists()
    title = None
    if not config_file_exist:
        logging.warning(f"Config file {args.config} does not exist")
    else:
        gin.parse_config_file(args.config)
        title = extract_specials_feature(args.config)

    eval = True if args.eval else False
    episode_time, total_timesteps, save_every_n_timesteps = init_params()
    nb_learnging_cycles = total_timesteps // save_every_n_timesteps
    model_file = args.model

    env = create_env(episode_time, title=title, learning_state=1.0)

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    model = init_model(env)

    if eval:
        env = model.get_env()
        model = PPO.load(model_file)
        obs = env.reset()
        total_reward = 0

        for _ in range(episode_time):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            logging.debug(f"{obs = }")
            total_reward += rewards

        logging.debug(f"Total reward: {total_reward}")

        env.close()
        return

    for i in range(nb_learnging_cycles):
        ls = i / nb_learnging_cycles
        n_env = create_env(episode_time, title=title, learning_state=ls)
        model.set_env(n_env)
        model.learn(
            total_timesteps=save_every_n_timesteps,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=args.log,
        )
        model.save(model_file)
        n_env.close()

    # vec_env = model.get_env()
    # model = PPO.load(model_file)

    # obs = vec_env.reset()

    # for _ in range(episode_time):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render()
    env.close()


if __name__ == "__main__":
    main()
