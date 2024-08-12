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
    # file with all the model
    parser.add_argument("model", type=str)
    # pour le logger le niveau
    parser.add_argument("-v", "--verbose", action="store_true")
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
    episode_time=400,
    total_timesteps=2_000_000,
    save_every_n_timesteps=10_000,
) -> tuple:
    return episode_time, total_timesteps, save_every_n_timesteps


def evaluate_model(model_path: Path) -> None:
    logging.info(f"Evaluating model {model_path}")
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
        return

    gin.parse_config_file(config_file)
    title = extract_specials_feature(config_file)
    episode_time, _, _ = init_params()

    env = create_env(episode_time, title=title, learning_state=1.0)

    model = init_model(env)

    env = model.get_env()
    model = PPO.load(model_file)
    obs = env.reset()
    total_reward = 0

    for _ in range(episode_time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        # logging.debug(f"{obs = }")
        total_reward += rewards

    logging.info(f"Total reward: {total_reward[0]}")

    env.close()


def main() -> None:
    args = parse_args()
    logging_setup(
        "PPO_evaluation.log", level=logging.DEBUG if args.verbose else logging.INFO
    )
    # check if the config file exists
    main_model_file = Path(args.model)
    is_model_file_exist = main_model_file.exists()
    title = None
    if not is_model_file_exist:
        logging.warning(f"Model file {main_model_file} does not exist")
        raise FileNotFoundError

    # list the directory in the main_model_file
    all_models_path = list(main_model_file.iterdir())
    logging.debug(f"{all_models_path = }")

    for model_path in all_models_path:
        evaluate_model(model_path)

    logging.info("All models have been evaluated")


if __name__ == "__main__":
    main()
