from stable_baselines3 import PPO
from pathlib import Path
import gymnasium as gym
import argparse
import gin
import numpy as np
import pandas as pd

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
import chime


def parse_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    # file with all the model
    parser.add_argument("model", type=str)
    # pour le logger le niveau
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")
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
            clean_line = line.split(".", maxsplit=1)[1]
            special_features = clean_line + special_features
        special_features = special_features.strip()
    return special_features


def list_active_loggers():
    """
    Affiche tous les loggers actifs.
    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        print(f"Logger: {logger.name}, Level: {logging.getLevelName(logger.level)}")


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
    render_mode: str = "human",
) -> gym.Env:
    env = gym.make(
        "CrowdSimCar-v2",
        render_mode=render_mode,
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
        verbose=0,
        tensorboard_log="runs",
        gamma=gamma,
        # learning_rate=linear_schedule(1e-3),
    )


def get_model_files(model_path: Path) -> tuple:
    config_file = list(model_path.glob("*.gin"))[0]
    is_config_file_exist = config_file.exists()
    if not is_config_file_exist:
        logging.warning(f"Config file {config_file} does not exist")

    model_file = list(model_path.glob("*.zip"))[0]
    is_model_file_exist = model_file.exists()
    if not is_model_file_exist:
        logging.warning(f"Model file {model_file} does not exist")

    tensorboard_log = list(model_path.glob("PPO*"))[0]
    is_tensorboard_log_exist = tensorboard_log.exists()
    if not is_tensorboard_log_exist:
        logging.warning(f"Tensorboard log {tensorboard_log} does not exist")

    if (
        not is_config_file_exist
        or not is_model_file_exist
        or not is_tensorboard_log_exist
    ):
        return None

    return config_file, model_file, tensorboard_log


@gin.configurable
def init_params(
    episode_time=400,
    total_timesteps=2_000_000,
    save_every_n_timesteps=10_000,
) -> tuple:
    return episode_time, total_timesteps, save_every_n_timesteps


def evaluate_model(config_file: Path, model_file: Path, render: True) -> tuple:

    gin.parse_config_file(config_file)
    title = extract_specials_feature(config_file)
    episode_time, _, _ = init_params()

    # just to have a different seed for each evaluation
    evaluation_seed = np.random.randint(0, 1000)
    np.random.seed(evaluation_seed)

    env = create_env(
        episode_time,
        title="Evaluate",
        learning_state=1.0,
        ghost_mode=True,
        nb_pedestrians=0,
        render_mode=None,
    )
    env.reset()
    maximum_reward = 0

    for _ in range(episode_time):
        action = env.unwrapped.robot.predict_what_to_do()
        obs, reward, _, _, _ = env.step(action)
        maximum_reward += reward

    logging.debug(f"Maximum reward: {maximum_reward:.2f}")
    env.close()

    np.random.seed(1001)
    np.random.seed(evaluation_seed)
    render_mode = "human" if render else None
    env = create_env(
        episode_time, title=title, learning_state=1.0, render_mode=render_mode
    )

    model = init_model(env)

    env = model.get_env()
    model = model.load(model_file)
    obs = env.reset()
    total_reward = 0
    collision_free_episode = True

    for _ in range(episode_time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if render:
            env.render()
        # logging.debug(f"{obs = }")
        if info[0]["info"] == "Collision" or info[0]["info"] == "GhostModeCollision":
            collision_free_episode = False
        total_reward += rewards

    logging.debug(f"Total reward: {total_reward[0]}")

    env.close()
    return maximum_reward, total_reward[0], collision_free_episode


def main() -> None:
    args = parse_args()
    chime.theme("pokemon")
    logging_setup(
        "PPO_evaluation.log", level=logging.DEBUG if args.verbose else logging.INFO
    )
    # check if the config file exists
    main_model_file = Path(args.model)
    is_model_file_exist = main_model_file.exists()
    if not is_model_file_exist:
        logging.warning(f"Model file {main_model_file} does not exist")
        chime.error()
        raise FileNotFoundError

    # list the directory in the main_model_file
    all_models_path = list(main_model_file.iterdir())
    # filter out the potential files
    all_models_path = [model for model in all_models_path if model.is_dir()]
    logging.debug(f"{all_models_path = }")
    do_render = args.render
    nb_reapeat = 100 if not do_render else 1
    df_results = pd.DataFrame(
        columns=[
            "model",
            "title",
            "maximum_reward",
            "model_cumulative_reward",
            "performance",
            "collision_free_episode",
        ]
    )

    for model_path in all_models_path:
        logging.info(f"Evaluating model {model_path}")
        maximum_reward_list = []
        model_cumulative_reward_list = []
        collision_free_episode_list = []

        core_paths = get_model_files(model_path)
        if core_paths is None:
            continue
        config_file, model_file, tensorboard_log = core_paths
        title = extract_specials_feature(config_file).replace("\n", ", ")
        logging.info(f"Title: {title}")

        progress_bar = tqdm(range(nb_reapeat), desc="Evaluating model", ncols=100)
        for _ in range(nb_reapeat):
            progress_bar.update(1)
            (
                maximum_reward,
                model_cumulative_reward,
                collision_free_episode,
            ) = evaluate_model(config_file, model_file, do_render)
            maximum_reward_list.append(maximum_reward)
            model_cumulative_reward_list.append(model_cumulative_reward)
            collision_free_episode_list.append(collision_free_episode)
        progress_bar.close()

        model_cumulative_reward_list = np.array(model_cumulative_reward_list)
        maximum_reward_list = np.array(maximum_reward_list)
        collision_free_episode_list = np.array(collision_free_episode_list)
        logging.info(f"Model: {model_path}")
        logging.info(f"Maximum reward: {maximum_reward_list.mean():.2f}")
        logging.info(
            f"Model cumulative reward: {model_cumulative_reward_list.mean():.2f}"
        )
        logging.info(
            f"Performance: {model_cumulative_reward_list.mean() / maximum_reward_list.mean():.2%}"
        )
        logging.info(
            f"Collision free episode: {collision_free_episode_list.mean():.2%}\n"
        )

        df_results.loc[len(df_results)] = {
            "model": model_path,
            "title": title,
            "maximum_reward": maximum_reward_list.mean(),
            "model_cumulative_reward": model_cumulative_reward_list.mean(),
            "performance": 100
            * model_cumulative_reward_list.mean()
            / maximum_reward_list.mean(),
            "collision_free_episode": collision_free_episode_list.mean() * 100,
        }

    # sort the dataframe by performance
    df_results = df_results.sort_values(by="performance", ascending=False)
    markdown_table = df_results.to_markdown(index=False)
    Console().print(Markdown(markdown_table))
    result_file = main_model_file / "results.md"
    with open(result_file, "w") as f:
        f.write(markdown_table)
        f.write("\n")
        # we add the gin config too
        f.write(f"```gin\n")
        with open(config_file, "r") as config:
            f.write(config.read())
        f.write(f"```")

    logging.info("All models have been evaluated")
    chime.success()


if __name__ == "__main__":
    main()
