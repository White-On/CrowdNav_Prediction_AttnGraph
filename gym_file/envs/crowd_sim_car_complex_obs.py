import gymnasium as gym
import gymnasium.spaces
import numpy as np

from env_component.human import Human
from env_component.robot import Robot
from env_component.agent import Agent, AgentGroup
import logging
import matplotlib.pyplot as plt
from gym_file.envs.crowd_sim_car import CrowdSimCar


class CrowdSimCarComplexObs(CrowdSimCar):
    """
    Environment for the crowd simulation with a car as the robot.
    """

    def __init__(
        self,
        render_mode=None,
        arena_size=6,
        nb_pedestrians=10,
        episode_time=100,
        time_step=0.1,
        display_future_trajectory=True,
        robot_is_visible=False,
        load_scenario=None,
        nb_goals_agent = 5,
        context_max_size=10,
    ):
        super().__init__(
            render_mode,
            arena_size,
            nb_pedestrians,
            episode_time,
            time_step,
            display_future_trajectory,
            robot_is_visible,
            load_scenario,
            nb_goals_agent,
            context_max_size=context_max_size,
        )

    def define_observations_space(
        self, forseen_index: int, nb_humans: int, nb_graph_feature: int
    ) -> gymnasium.spaces.Dict:
        observation_space = {}
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        # vehicle_speed_boundries = [-0.5, 2]
        # vehicle_angle_boundries = [-np.pi/6, np.pi/6]
        # objectives_boundries = np.full((forseen_index, 2), [-10,10])
        # all_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries, objectives_boundries))
        # observation_space['robot_node'] = gymnasium.spaces.Box(low= all_boundries[:,0], high=all_boundries[:,1], dtype=np.float32)
        # current position -> 2 coordinates
        # goal position -> 2 coordinates * forseen_index <- how many steps we want to see in the future
        observation_space["robot_node"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + forseen_index * 2,), dtype=np.float32
        )

        # predictions only include mu_x, mu_y (or px, py)
        spatial_edge_dim = int(2 * (nb_graph_feature))

        # Here this should be a graph to go inside the GNN but for now we will use a matrix
        # To make sure there wont be any overload, we use the context_max_size
        observation_space["graph_features"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.context_max_size, spatial_edge_dim),
            dtype=np.float32,
        )
        logging.debug(f'{observation_space}')
        return gymnasium.spaces.Dict(observation_space)

    def generate_observation(self) -> dict:
        """Generate observation for reset and step functions"""

        observation = {}
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        observation["robot_node"] = np.array(self.robot.get_robot_state())

        # graph features: future position of every human + robot
        # dim = [num_visible_humans + 1, 2*(self.predict_steps+1)]
        placeholder_value = 0.0
        observation["graph_features"] = np.full(
            (self.context_max_size, (self.nb_time_steps_seen_as_graph_feature), 2), placeholder_value
        )

        visible_agent_by_robot = agent_visible.filter(
            lambda x: x.id != self.robot.id
        ).filter(self.robot.can_i_see)

        # amoung the visible agent, we take only the context_max_size closest agents
        visible_agent_by_robot = visible_agent_by_robot.sort(
            lambda x: self.distance_matrix[self.robot.id][x.id]
        ).limit(self.context_max_size
        ).get_all()

        # transform the graph features into relative coordinates
        robot_position = np.array(self.robot.get_position())
        robot_rotation = self.robot.orientation

        for i, human in enumerate(visible_agent_by_robot):
            direction_vector = np.array(human.speed)
            direction_vector *= self.time_step
            # with the vector we calculate the n future positions
            human_position = np.array(human.get_position())
            direction_vector = np.tile(
                direction_vector, self.nb_time_steps_seen_as_graph_feature
            ).reshape(-1, 2) * np.arange(
                0, self.nb_time_steps_seen_as_graph_feature
            ).reshape(
                -1, 1
            )
            human_future_traj = human_position + direction_vector
            observation["graph_features"][i] = self.global_to_relative(
                human_future_traj.reshape(-1, 2), robot_position, robot_rotation
            )
        observation["graph_features"] = observation["graph_features"].reshape(self.context_max_size, -1)
        # logging.info(f'{observation["graph_features"].shape}')
        # logging.debug(f"🔵 observation: {observation}")
        return observation
