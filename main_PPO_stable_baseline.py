from stable_baselines3 import PPO
import gymnasium as gym
import argparse
import gin

from gym_file.envs.crowd_sim_car import CrowdSimCar
from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
import logging
from logger import logging_setup

def parse_args():
     # parse arguments
    parser = argparse.ArgumentParser()
    # mode eval ou pas
    parser.add_argument("-e", "--eval", action='store_true')
    # pour le logger le niveau
    parser.add_argument("-v", "--verbose", action='store_true')
    # nom fichier log tensorflow
    parser.add_argument("-l", "--log", type=str, default="PPO")
    # nom de fichier de configuration gin
    parser.add_argument("-c", "--config", type=str, default="config.gin")
    # nom du fichier du model sauvegardé
    parser.add_argument("-m", "--model", type=str, default="ppo_CrowdSimCar")
    return parser.parse_args()

@gin.configurable
def create_env(episode_time: int,
               nb_pedestrians: int = 10,
               robot_is_visible: bool = True,
               nb_goals_agent: int = 5,
               scenario: str = None,):
    env = gym.make(
        "CrowdSimCar-v0",
        render_mode="human",
        episode_time=episode_time,
        nb_pedestrians=nb_pedestrians,
        disable_env_checker=True,
        load_scenario=scenario,
        robot_is_visible=robot_is_visible,
        nb_goals_agent = nb_goals_agent,
    )
    return env

@gin.configurable(denylist=["env"])
def init_model(env,
            gamma: float = 0.99,
               ):
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
        gamma=gamma,
        # learning_rate=linear_schedule(1e-3),
    )


def main():
    args = parse_args()
    logging_setup("PPO_evaluation.log", level=logging.DEBUG if args.verbose else logging.INFO)
    gin.parse_config_file(args.config)
    episode_time = 200
    eval = True if args.eval else False
    total_timesteps = 2_000_000
    save_every_n_timesteps = 10_000
    nb_learnging_cycles = total_timesteps // save_every_n_timesteps
    model_file = args.model

    env = create_env(episode_time)

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
            logging.info(f"{obs['robot_node'] = }") 
            total_reward += rewards

        logging.info(f"Total reward: {total_reward}")
        
        return

    for i in range(nb_learnging_cycles):
        model.learn(
            total_timesteps=save_every_n_timesteps,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=args.log,
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
