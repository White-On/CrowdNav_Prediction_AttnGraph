import gym
import gym.spaces
import numpy as np

from env_component.human import Human
from env_component.robot import Robot
from env_component.agent import Agent, AgentGroup
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CrowdSimCar(gym.Env):
    """
    Environment for the crowd simulation with a robot and pedestrians
    """

    metadata = {"render_modes": ["human", "debug", None]}

    def __init__(
        self,
        render_mode=None,
        arena_size=6,
        nb_pedestrians=10,
        episode_time=100,
        time_step=0.1,
        display_future_trajectory=True,
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

        self.goal_threshold_distance = self.robot.radius

        self.observation_space = self.define_observations_space(
            self.robot.nb_forseen_goal,
            nb_humans=nb_pedestrians,
            nb_graph_feature=self.nb_time_steps_seen_as_graph_feature,
        )
        self.action_space = self.define_action_space()
        self.all_agent_group = AgentGroup(*Agent.ENTITIES)

    def define_observations_space(
        self, forseen_index: int, nb_humans: int, nb_graph_feature: int
    ) -> gym.spaces.Dict:
        observation_space = {}
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        # vehicle_speed_boundries = [-0.5, 2]
        # vehicle_angle_boundries = [-np.pi/6, np.pi/6]
        # objectives_boundries = np.full((forseen_index, 2), [-10,10])
        # all_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries, objectives_boundries))
        # observation_space['robot_node'] = gym.spaces.Box(low= all_boundries[:,0], high=all_boundries[:,1], dtype=np.float32)
        observation_space["robot_node"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + forseen_index * 2,), dtype=np.float32
        )

        # predictions only include mu_x, mu_y (or px, py)
        spatial_edge_dim = int(2 * (nb_graph_feature))

        observation_space["graph_features"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(nb_humans, spatial_edge_dim),
            dtype=np.float32,
        )

        observation_space["visible_masks"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(nb_humans,), dtype=np.float32
        )

        return gym.spaces.Dict(observation_space)

    def define_action_space(self) -> gym.spaces.Box:
        # vehicle_speed_boundries = [-0.5, 2]
        vehicle_speed_boundries = [-0.2, 0.2]
        # TODO put back the ability to drive backward
        # vehicle_speed_boundries = [0.0, 2.0]
        # True limit angle is 14 degrees but for now we will use 30 degrees
        # limit_angle = np.deg2rad(14)
        limit_angle = np.pi / 6
        vehicle_angle_boundries = [-limit_angle, limit_angle]

        action_space_boundries = np.vstack(
            (vehicle_speed_boundries, vehicle_angle_boundries)
        )
        return gym.spaces.Box(
            action_space_boundries[:, 0], action_space_boundries[:, 1], dtype=np.float32
        )

    def reset(self) -> dict:
        """
        Reset the environment
        :return:
        """

        self.global_time = 0
        self.all_agent_group.reset()
        if len(Human.HUMAN_LIST) != 0:
            distance_from_human = self.robot.distance_from_other_agents(
                [human.get_position() for human in Human.HUMAN_LIST]
            )
            while self.compute_collision_reward(distance_from_human) < 0:
                self.robot.reset()
                distance_from_human = self.robot.distance_from_other_agents(
                    [human.get_position() for human in Human.HUMAN_LIST]
                )

        # get robot observation
        observation_after_reset = self.generate_observation()

        return observation_after_reset

    def step(self, robot_action: np.ndarray) -> tuple:
        """
        step function
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        robot_action = np.clip(
            robot_action, self.action_space.low, self.action_space.high
        )

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward()

        if done:
            self.all_agent_group.reset()

        # apply action and update all agents
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)
        # TODO optimize this part to avoid double computation

        # all_agent_coordinates = np.array(agent_visible.apply(lambda x: x.coordinates))
        # matrix_distance_from_each_other = np.linalg.norm(all_agent_coordinates[:, None] - all_agent_coordinates, axis=2) 
        # all_sensor_range = np.array(agent_visible.apply(lambda x: x.sensor_range))
        # can_see_each_other = matrix_distance_from_each_other < all_sensor_range[:, None] + all_sensor_range
        
        # Human step
        for human in Human.HUMAN_LIST:
            other_agent_state = (
                agent_visible.filter(lambda x: x.id != human.id)
                .filter(human.can_i_see)
                .apply(lambda x: x.coordinates + x.speed)
            )
            # other_agent_state = (
            #     agent_visible.filter(lambda x: x.id != human.id)
            #     .filter(lambda x: can_see_each_other[human.id][x.id])
            #     .apply(lambda x: x.coordinates + x.speed)
            # )
            # predict what to do
            human_action = human.predict_what_to_do(*other_agent_state)
            human.step(human_action)
            if human.is_goal_reached(self.goal_threshold_distance):
                human.goal_coordinates = human.set_random_goal()

        # Robot step
        self.robot.step(robot_action)

        # I put this in reward calculation
        # is_robot_reach_goal = self.robot.is_goal_reached(self.goal_threshold_distance)
        # if is_robot_reach_goal:
        #     self.robot.next_goal()
        # all_goals_reached = self.robot.current_goal_cusor >= len(self.robot.collection_goal_coordinates)
        # if all_goals_reached:
        #     logging.info('All robot goals are reached!')
        #     self.all_agent_group.reset()

        self.global_time += 1

        # compute the observation
        step_observation = self.generate_observation()

        return step_observation, reward, done, episode_info

    def generate_observation(self) -> dict:
        """Generate observation for reset and step functions"""

        observation = {}
        nb_humans_in_simulation = len(Human.HUMAN_LIST)
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        observation["robot_node"] = np.array(self.robot.get_robot_state())

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
        # TODO/WARNING: Giving the robot relative position of the robot itself
        # does not make sense
        robot_position = np.array(self.robot.get_position())
        # add robot future traj
        # robot_future_traj = np.tile(self.robot.speed, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 2) * np.arange(0, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 1)
        # observation['graph_features'][-1] = robot_future_traj
        observation["graph_features"] = observation["graph_features"] - robot_position
        observation["graph_features"] = observation["graph_features"].reshape(
            nb_humans_in_simulation, -1
        )

        list_of_visible_humans = visible_agent_by_robot.apply(lambda x: x.id)
        visibility_mask = [
            True if human.id in list_of_visible_humans else False
            for human in Human.HUMAN_LIST
        ]
        observation["visible_masks"] = visibility_mask

        return observation

    def compute_collision_reward(self, distance_from_human: float) -> float:
        distance_limit = self.robot.radius + Human.HUMAN_LIST[0].radius

        collision_happed = np.min(distance_from_human) < distance_limit

        if collision_happed:
            return -100.0
        else:
            return 0.0

    def compute_near_collision_reward(self, distance_from_human: float) -> float:
        min_distance_to_keep_from_human = 1.5
        distance_to_closest_human = np.min(distance_from_human)
        vehicle_current_speed = self.robot.velocity_norm
        vehicle_min_acceleration = self.robot.acceleration_limits[0]

        dr = np.max(
            [
                min_distance_to_keep_from_human,
                (vehicle_current_speed**2.0) / (2.0 * vehicle_min_acceleration),
            ]
        )

        return np.exp((distance_to_closest_human - dr) / dr)

    # OLD FORMULA
    # def compute_speed_reward(self,current_speed:float, pref_speed:float)->float:
    #     if 0.0 < current_speed <= pref_speed:
    #         # l = 1/pref_speed # old formula
    #         # return l * (pref_speed - current_speed)
    #         return 1-(pref_speed - current_speed)/pref_speed
    #     elif current_speed > pref_speed:
    #         return np.exp(-current_speed + pref_speed)
    #     elif current_speed <= 0.0:
    #         return current_speed

    def compute_speed_reward(self, current_speed: float, pref_speed: float) -> float:
        if current_speed <= pref_speed:
            return (np.exp(current_speed - pref_speed) - 0.5) * 2
        elif current_speed > pref_speed:
            return (np.exp(-current_speed + pref_speed) - 0.5) * 2

    def compute_angular_reward(self, angle: float) -> float:
        # TODO Careful with hard coded values
        angle_penalty = 20
        return (np.exp(-angle / angle_penalty) - 0.5) * 2

    def compute_proximity_reward(self, distance_from_goal: float) -> float:
        # TODO Careful with hard coded values
        # penalty_distance = 1
        # return 1 - 2 / (1 + np.exp(0.5*(-distance_from_goal + penalty_distance)))
        return 1 - 2 / (1 + np.exp((-distance_from_goal)))

    def calc_reward(self, save_in_file=False) -> tuple:
        if len(Human.HUMAN_LIST) != 0:
            distance_from_human = self.robot.distance_from_other_agents(
                [human.get_position() for human in Human.HUMAN_LIST]
            )

            collision_reward = self.compute_collision_reward(distance_from_human)
            near_collision_reward = self.compute_near_collision_reward(
                distance_from_human
            )
        else:
            collision_reward = 0
            near_collision_reward = 0

        speed_reward = self.compute_speed_reward(
            self.robot.velocity_norm, self.robot.desired_speed
        )
        angle_from_goal = np.abs(self.robot.get_angle_from_goal())
        # print(f'angle_from_goal: {np.degrees(angle_from_goal)}')
        angular_reward = self.compute_angular_reward(np.degrees(angle_from_goal))

        # current_goal_coordinates = self.robot.get_current_visible_goal()
        # if current_goal_coordinates is None:
        #     distance_from_goal = 0
        # else:
        #     distance_from_goal = np.linalg.norm(np.array(self.robot.coordinates) - np.array(current_goal_coordinates))
        distance_from_path = self.robot.get_distance_from_path()
        proximity_reward = self.compute_proximity_reward(distance_from_path)

        collision_factor = 1
        near_collision_factor = 0
        speed_factor = 6
        angular_factor = 2
        proximity_factor = 3

        collision_reward *= collision_factor
        near_collision_reward *= near_collision_factor
        speed_reward *= speed_factor
        angular_reward *= angular_factor
        proximity_reward *= proximity_factor

        reward = (
            collision_reward
            + near_collision_reward
            + speed_reward
            + angular_reward
            + proximity_reward
        )

        logging.debug(
            f"💥collision_reward: {collision_reward:>7.2f},\n\
                    🚸 near_collision_reward: {near_collision_reward:>7.2f},\n\
                    🚀 speed_reward: {speed_reward:>7.2f},\n\
                    📐 angular_reward: {angular_reward:>7.2f},\n\
                    🤏 proximity_reward: {proximity_reward:>7.2f},\n\
                    🏆 reward: {reward:>7.2f}"
        )

        if save_in_file:
            with open("reward.csv", "a") as f:
                f.write(
                    f"{collision_reward},{near_collision_reward},{speed_reward},{angular_reward},{proximity_reward},{reward}\n"
                )

        episode_timeout = self.global_time >= self.episode_time - 1
        collision_happened = collision_reward < 0
        reward_all_goals_reached = 50
        reward_single_goal_reached = 20

        is_robot_reach_goal = self.robot.is_goal_reached(self.goal_threshold_distance)
        if is_robot_reach_goal:
            reward += reward_single_goal_reached
            self.robot.next_goal()
        all_goals_reached = self.robot.current_goal_cusor >= len(
            self.robot.collection_goal_coordinates
        )
        if all_goals_reached:
            logging.debug("All robot goals are reached!")
            reward += reward_all_goals_reached
            self.all_agent_group.reset()

        # logging.debug(f'🎯 distance_from_goal: {is_robot_reach_goal:>7.2f}, 🎯 goal_distance_threshold: {goal_distance_threshold:>7.2f}, 🎯 goal_reached: {goal_reached:>7.2f}')
        logging.debug(f"🎯 distance_from_goal: {is_robot_reach_goal:>7.2f}")

        conditions = {
            episode_timeout: "Timeout",
            collision_happened: "Collision",
            all_goals_reached: "ReachGoal",
        }

        for condition, result in conditions.items():
            if condition:
                done = True
                episode_info = result
                break
        else:
            done = False
            episode_info = "Nothing"

        return reward, done, episode_info

    def _render_frame(self) -> None:
        """
        render function
        """
        if self.render_mode == None:
            return

        robot_color = "gold"
        human_color = "gray"
        visible_human_color = "blue"
        robot_goal_color = "green"
        human_goal_color = "lightblue"
        direction_arrow_color = "black"
        color_robot_path = robot_goal_color
        robot_visual_radius = 10
        human_visual_radius = 6

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array(
                [
                    [np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1],
                ]
            )
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        ax.clear()
        ax.set_xlim(-self.arena_size, self.arena_size)
        ax.set_ylim(-self.arena_size, self.arena_size)

        robotX, robotY = self.robot.coordinates
        robot_radius = self.robot.radius

        # add list of goals
        for idx, goal in enumerate(self.robot.collection_goal_coordinates):
            color = robot_goal_color
            markersize_goal = 10
            if idx == self.robot.current_goal_cusor:
                color = "r"
            ax.plot(
                goal[0],
                goal[1],
                color=color,
                marker="p",
                markersize=markersize_goal,
                label="Goal",
            )
            offset_txt = 0.05
            plt.text(
                goal[0] - offset_txt,
                goal[1] - offset_txt,
                idx,
                color="black",
                fontsize=markersize_goal - 1,
            )

        # add line for the path between goals
        if len(self.robot.path) > 0:
            for path in self.robot.path:
                ax.plot(
                    [path[0], path[2]],
                    [path[1], path[3]],
                    color=color_robot_path,
                    linestyle="--",
                )

        # add robot
        # ax.plot(*self.robot.get_position(), color=robot_color, marker='o', markersize=robot_visual_radius, label='Robot')
        ax.add_artist(
            patches.Rectangle(
                (robotX - robot_radius, robotY - robot_radius),
                3 * robot_radius,
                2 * robot_radius,
                color=robot_color,
                linewidth=0.5,
                angle=np.degrees(self.robot.orientation),
                rotation_point=(robotX, robotY),
            )
        )
        # direction and goal arrow
        if self.render_mode == "debug":
            ax.arrow(
                x=self.robot.coordinates[0],
                y=self.robot.coordinates[1],
                dx=self.robot.speed[0],
                dy=self.robot.speed[1],
                head_width=0.1,
                head_length=0.1,
                fc=direction_arrow_color,
                ec=direction_arrow_color,
            )
            velocity_toward_goal = np.array(
                self.robot.get_current_visible_goal()[0]
            ) - np.array(self.robot.coordinates)
            normalized_velocity_toward_goal = velocity_toward_goal / np.linalg.norm(
                velocity_toward_goal
            )
            ax.arrow(
                x=self.robot.coordinates[0],
                y=self.robot.coordinates[1],
                dx=normalized_velocity_toward_goal[0],
                dy=normalized_velocity_toward_goal[1],
                head_width=0.1,
                head_length=0.1,
                fc=robot_goal_color,
                ec=robot_goal_color,
            )

            ax.arrow(
                x=self.robot.coordinates[0],
                y=self.robot.coordinates[1],
                dx=self.robot.acceleration[0],
                dy=self.robot.acceleration[1],
                head_width=0.1,
                head_length=0.1,
                fc="r",
                ec="r",
            )

        # draw FOV for the robot
        # TODO if the robot has a FOV diff than 360

        # add an arc of robot's sensor range
        ax.add_artist(
            plt.Circle(
                (robotX, robotY),
                self.robot.sensor_range,
                fill=False,
                color="black",
                linestyle="--",
                linewidth=0.5,
            )
        )

        id_visible_agent_by_robot = (
            self.all_agent_group.filter(lambda x: x.id != self.robot.id)
            .filter(self.robot.can_i_see)
            .apply(lambda x: x.id)
        )
        # add human
        for human in Human.HUMAN_LIST:
            color = human_color
            if human.id in id_visible_agent_by_robot:
                color = visible_human_color
            ax.plot(
                *human.get_position(),
                color=color,
                marker="o",
                markersize=human_visual_radius,
                label="Human",
            )
            if self.render_mode == "debug":
                ax.plot(
                    *human.goal_coordinates,
                    color=human_goal_color,
                    marker="o",
                    label="Human Goal",
                )
                ax.arrow(
                    x=human.coordinates[0],
                    y=human.coordinates[1],
                    dx=human.speed[0],
                    dy=human.speed[1],
                    head_width=0.1,
                    head_length=0.1,
                    fc=direction_arrow_color,
                    ec=direction_arrow_color,
                )
                velocity_toward_goal = np.array(human.goal_coordinates) - np.array(
                    human.coordinates
                )
                normalized_velocity_toward_goal = velocity_toward_goal / np.linalg.norm(
                    velocity_toward_goal
                )
                ax.arrow(
                    x=human.coordinates[0],
                    y=human.coordinates[1],
                    dx=normalized_velocity_toward_goal[0],
                    dy=normalized_velocity_toward_goal[1],
                    head_width=0.1,
                    head_length=0.1,
                    fc=human_goal_color,
                    ec=human_goal_color,
                )

        if self.display_future_trajectory:
            observation = self.generate_observation()
            predicted_positions = observation["graph_features"]
            visible_masks = observation["visible_masks"]
            # we remove the predicted positions with the visibility mask
            predicted_positions = predicted_positions[visible_masks]
            for human_future_traj in predicted_positions:
                human_future_traj = human_future_traj.reshape(-1, 2)
                human_future_traj = human_future_traj + np.array([robotX, robotY])
                ax.plot(
                    human_future_traj[:, 0],
                    human_future_traj[:, 1],
                    color="tab:orange",
                    marker="o",
                    markersize=human_visual_radius,
                    label="Human Future Traj",
                    alpha=0.5,
                    linestyle="--",
                )

        # plot reward space for experiment
        # x = np.meshgrid(np.linspace(-self.arena_size, self.arena_size, 5), np.linspace(-self.arena_size, self.arena_size, 5))
        # reward_color = 'red'
        # plt.scatter(x[0], x[1], color=reward_color)

        if self.render_delay:
            plt.pause(self.render_delay)
        else:
            plt.pause(0.0001)

    def render(self) -> None:
        if self.render_mode == None:
            return
        self._render_frame()


def distance_from_line(x: float, y: float, path: list) -> float:
    position = [x, y]
    path = np.array(path)

    if np.array_equal(position, path[0]):
        return 0.0

    a = position - path[0]
    b = path[1] - path[0]
    d = np.linalg.norm(a) * np.cos(
        np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    )
    normal_point = path[0] + d * b / np.linalg.norm(b)
    distance_to_path = np.linalg.norm(position - normal_point)
    # print(f"Distance to path: {distance_to_path}")
    return distance_to_path
