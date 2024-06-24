import pytest
import numpy as np
import gym

from env_component.human import Human
from env_component.robot import Robot
from env_component.agent import Agent, AgentGroup

from gym_file.envs.crowd_sim_car_simple_obs import CrowdSimCarSimpleObs
from gym_file.envs.crowd_sim_car import CrowdSimCar


@pytest.fixture()
def random_agent():
    agent = Agent()
    return agent


@pytest.fixture(
    params=np.arange(0, 30, 5), ids=[f"amount_of_agents={i}" for i in range(0, 30, 5)]
)
def random_group_human(request):
    for _ in range(request.param):
        Human(delta_t=0.1, is_visible=True, radius=0.2)
    return Human.HUMAN_LIST


@pytest.fixture()
def random_robot():
    robot = Robot(delta_t=0.1, radius=0.2, is_visible=True)
    return robot


def test_human_generation(random_group_human):
    Human.apply(Human.reset)
    assert Human.apply(lambda x: x.coordinates != [None, None]) == [True] * len(
        random_group_human
    )


def test_robot_generation(random_robot):
    robot = random_robot
    robot.reset()
    assert robot.coordinates != [None, None]


# def test_run_env():
#     env = gym.make(
#         "CrowdSimCar-v0",
#         render_mode=None,
#         episode_time=200,
#         nb_pedestrians=20,
#         disable_env_checker=True,
#         robot_is_visible=True,
#     )

#     env.reset()
#     for step in range(200):
#         action = env.robot.predict_what_to_do()
#         obs, reward, done, info = env.step(action)
#         if done:
#             break

#         assert isinstance(obs, dict)
#         assert isinstance(reward, float)
#         assert isinstance(done, bool)
