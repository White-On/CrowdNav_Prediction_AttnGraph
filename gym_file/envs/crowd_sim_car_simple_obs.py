import gym
import gym.spaces
import numpy as np

from human import Human
from robot import Robot
from agent import Agent, AgentGroup
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym_file.envs.crowd_sim_car import CrowdSimCar

from rich import print


class CrowdSimCarSimpleObs(CrowdSimCar):
    """
    Environment for the crowd simulation with a robot and pedestrians.
    """

    metadata = {"render_modes": ["human", "debug", None]}

    def __init__(
        self,
        render_mode=None,
        arena_size=6,
        nb_pedestrians=10,
        episode_time=100,
        time_step=0.1,
        display_future_trajectory=False,
        robot_is_visible=False,
    ):
        self.arena_size = arena_size
        if render_mode not in self.metadata["render_modes"]:
            logging.error(f"Mode {render_mode} is not supported")
            raise NotImplementedError
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-arena_size, arena_size)
            ax.set_ylim(-arena_size, arena_size)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.ion()
            plt.show()
            self.render_axis = ax
            # self.render_delay = 1.0
            self.render_delay = None
            self.display_future_trajectory = display_future_trajectory

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.episode_time = episode_time
        self.time_step = time_step
        self.nb_time_steps_seen_as_graph_feature = 5
        self.nb_forseen_goal = 1
        self.goal_threshold_distance = 0.3

        sensor_range = 4
        self.robot = Robot(
            self.time_step,
            arena_size=arena_size,
            sensor_range=sensor_range,
            nb_forseen_goal=self.nb_forseen_goal,
            is_visible=robot_is_visible,
            radius=0.3,
            nb_goals=5,
        )
        for _ in range(nb_pedestrians):
            Human(self.time_step, arena_size=arena_size, sensor_range=sensor_range)

        self.observation_space = self.define_observations_space(
            self.robot.nb_forseen_goal,
            nb_humans=nb_pedestrians,
            nb_graph_feature=self.nb_time_steps_seen_as_graph_feature,
        )
        self.action_space = self.define_action_space()
        self.all_agent_group = AgentGroup(*Agent.ENTITIES)

    def define_observations_space(
        self, forseen_index: int, nb_humans: int, nb_graph_feature: int
    ) -> gym.spaces.Box:
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        # predictions only include mu_x, mu_y (or px, py)
        spatial_edge_dim = int(2 * (nb_graph_feature))

        robot_node_shape = 1 + 1 + forseen_index * 2
        graph_feature_shape = nb_humans * spatial_edge_dim

        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(robot_node_shape + graph_feature_shape,),
            dtype=np.float32,
        )

    def generate_observation(self) -> list:
        """Generate observation for reset and step functions"""

        observation = {}
        nb_humans_in_simulation = len(Human.HUMAN_LIST)
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        observation["robot_node"] = np.array(self.robot.get_robot_state())
        if nb_humans_in_simulation == 0:
            return observation["robot_node"].flatten().tolist()

        # graph features: future position of every human + robot
        # dim = [num_visible_humans + 1, 2*(self.predict_steps+1)]
        observation["graph_features"] = np.zeros(
            (nb_humans_in_simulation, (self.nb_time_steps_seen_as_graph_feature), 2)
        )
        # logging.info(f"graph_features: {observation['graph_features'].shape}")

        visible_agent_by_robot = agent_visible.filter(
            lambda x: x.id != self.robot.id
        ).filter(self.robot.can_i_see)

        for i, human in enumerate(Human.HUMAN_LIST):
            if human.id in visible_agent_by_robot.apply(lambda x: x.id):
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
            else:
                # the visibility mask will make sure that the invisible humans are not considered
                human_future_traj = np.zeros(
                    (self.nb_time_steps_seen_as_graph_feature, 2)
                )
            observation["graph_features"][i] = human_future_traj

        # transform the graph features into relative coordinates
        # TODO/WARNING: Giving the robot relative position to the robot itself
        # does not make sense
        robot_position = np.array(self.robot.get_position())
        # add robot future traj
        # robot_future_traj = np.tile(self.robot.speed, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 2) * np.arange(0, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 1)
        # observation['graph_features'][-1] = robot_future_traj
        observation["graph_features"] = observation["graph_features"] - robot_position
        robot_rotation = self.robot.orientation
        # Rotate the translated coordinates by the negative of the point's orientation
        rotation_matrix = np.array(
            [
                [np.cos(-robot_rotation), -np.sin(-robot_rotation)],
                [np.sin(-robot_rotation), np.cos(-robot_rotation)],
            ]
        )
        observation["graph_features"] = np.dot(
            observation["graph_features"], rotation_matrix.T
        )
        observation["graph_features"] = observation["graph_features"].reshape(
            nb_humans_in_simulation, -1
        )

        list_of_visible_humans = visible_agent_by_robot.apply(lambda x: x.id)
        visibility_mask = [
            True if human.id in list_of_visible_humans else False
            for human in Human.HUMAN_LIST
        ]
        observation["visible_masks"] = visibility_mask
        flatten_observation = np.concatenate(
            (
                observation["robot_node"].flatten(),
                observation["graph_features"].flatten(),
            ),
            axis=None,
        )
        return flatten_observation.tolist()
