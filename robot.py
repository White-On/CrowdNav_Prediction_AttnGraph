from agent import Agent
import numpy as np
import logging


class Robot(Agent):
    def __init__(
        self,
        delta_t: float,
        nb_goals=5,
        nb_forseen_goal=2,
        is_visible=True,
        desired_speed=1.0,
        radius=0.2,
        FOV=np.pi,
        **kwargs,
    ):
        super().__init__(is_visible, desired_speed, radius, FOV, delta_t, **kwargs)

        self.nb_goals = nb_goals
        self.collection_goal_coordinates = [None] * self.nb_goals
        self.path = None
        self.current_goal_cusor = 0
        self.velocity_norm = None
        self.speed_limit = [0.0, desired_speed*2]
        self.acceleration_limits = [-0.5, 0.5]
        self.acceleration = [0.0, 0.0]
        self.robot_size = 0.3
        self.orientation = 0
        self.theta = 0
        self.theta_acceleration = 0
        self.nb_forseen_goal = nb_forseen_goal

    def __str__(self) -> str:
        return f"{super().__str__()}, {self.id = }"

    def __repr__(self) -> str:
        return self.__str__()

    def set_random_position(self) -> None:
        position_limit = self.arena_size
        return [
            np.random.uniform(-position_limit, position_limit),
            np.random.uniform(-position_limit, position_limit),
        ]

    def set_random_speed(self) -> None:
        return [
            np.random.uniform(-self.desired_speed, self.desired_speed),
            np.random.uniform(-self.desired_speed, self.desired_speed),
        ]

    def set_random_goal(self) -> None:
        goal_coordinate_limit = self.arena_size
        return [
            [
                np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit),
                np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit),
            ]
            for _ in range(self.nb_goals)
        ]

    def create_path(self) -> list:
        # the path is just a collection of line segments between the goals and the initial position
        initial_position_robot = self.get_position()
        goals_coordinates = self.collection_goal_coordinates
        no_initial_position = initial_position_robot == [None, None]
        no_goals = goals_coordinates == [None] * self.nb_goals
        if no_initial_position or no_goals:
            logging.warn(
                f"Warning: {self.__class__.__name__} has no initial position or goals set!"
            )
            return None

        path_element = np.vstack((initial_position_robot, goals_coordinates))

        path = np.zeros((len(path_element) - 1, 4))

        for idx in range(len(path_element) - 1):
            path[idx] = np.array(
                [
                    path_element[idx][0],
                    path_element[idx][1],
                    path_element[idx + 1][0],
                    path_element[idx + 1][1],
                ]
            )

        # print(f"Path: {path}")
        # print(f"Goal: {goals_coordinates}")
        return path.tolist()
    
    def reset(self) -> None:
        self.coordinates = self.set_random_position()
        self.speed = self.set_random_speed()
        self.collection_goal_coordinates = self.set_random_goal()
        self.path = self.create_path()
        self.current_goal_cusor = 0
        self.velocity_norm = np.linalg.norm(self.speed)
        self.orientation = np.random.uniform(0, 2 * np.pi)

    def get_distance_from_path(self):
        position = self.get_position()
        idx = self.current_goal_cusor
        path = np.array(self.path[idx]).reshape(2, 2)

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

    

    def predict_what_to_do(self, *other_agent_state: list) -> list:
        acceleration_action = 0.0
        theta_action = self.get_angle_from_goal()
        # theta_action = np.pi/6
        logging.info(np.degrees(theta_action))
        return [acceleration_action, theta_action]

    def get_current_visible_goal(self) -> list:
        goal_cursor_too_far = self.current_goal_cusor >= len(
            self.collection_goal_coordinates
        )
        if goal_cursor_too_far:
            return None
        return self.collection_goal_coordinates[
            self.current_goal_cusor : self.current_goal_cusor + self.nb_forseen_goal
        ]

    def next_goal(self) -> None:
        self.current_goal_cusor += 1

    def is_goal_reached(self, threashold) -> bool:
        current_goal = self.get_current_visible_goal()
        if current_goal is None:
            logging.info(f"all goals have been resolved")
        current_goal = current_goal[0]
        return (
            np.linalg.norm(np.array(self.coordinates) - np.array(current_goal))
            < threashold
        )

    def step(self, action: list) -> None:
        """Action is a list with the acceleration and wheel orientation theta"""
        desired_acceleration, desired_theta = action
        self.theta = self.limit_theta_change(desired_theta)
        self.acceleration = self.limit_acceleration_change(desired_acceleration)
        self.speed += self.acceleration
        self.speed = self.limit_speed(self.speed)
        self.velocity_norm = np.linalg.norm(self.speed)
        self.orientation += self.compute_orientation()
        self.coordinates = self.compute_position()
        

    def limit_theta_change(self, desired_theta: float) -> float:
        current_theta = self.theta
        rad_change_limit = np.pi / 32

        lower_limit = current_theta - rad_change_limit
        upper_limit = current_theta + rad_change_limit

        return np.clip(desired_theta, lower_limit, upper_limit)

    def limit_acceleration_change(self, desired_acceleration: float) -> np.array:
        # same comment as the previous method
        clipped_acceleration_norm = np.clip(desired_acceleration, self.acceleration_limits[0], self.acceleration_limits[1])
        # we take the direction of the acceleration
        vector_direction = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        return clipped_acceleration_norm * vector_direction
    
    def limit_speed(self, speed: np.array) -> list:
        # clip the speed norm between the speed limits
        speed_norm = np.linalg.norm(speed)
        clipped_speed_norm = np.clip(speed_norm, self.speed_limit[0], self.speed_limit[1])
        # we take the direction of the speed
        normalized_speed = speed / speed_norm
        return (clipped_speed_norm * normalized_speed).tolist()


    def compute_orientation(self) -> float:
        orientation = (
            (self.velocity_norm / self.robot_size) * np.tan(self.theta) * self.delta_t
        )
        return orientation

    def compute_position(self) -> list:
        x = (
            self.velocity_norm * np.cos(self.orientation) * self.delta_t
            + self.coordinates[0]
        )
        y = (
            self.velocity_norm * np.sin(self.orientation) * self.delta_t
            + self.coordinates[1]
        )
        return [x, y]

        # self.x += v * np.cos(self.theta) * dt
        # self.y += v * np.sin(self.theta) * dt
        # self.theta += (v / self.L) * np.tan(delta) * dt

    def compute_speed_vector(self) -> list:
        v_x = self.velocity_norm * np.cos(self.orientation)
        v_y = self.velocity_norm * np.sin(self.orientation)
        return [v_x, v_y]

    def get_angle_from_goal(self) -> float:
        goal_coordinate = self.get_current_visible_goal()
        if goal_coordinate is None or len(goal_coordinate) == 0:
            return 0.0
        else:
            goal_coordinate = goal_coordinate[0]
        orientation = self.orientation % (2 * np.pi)
        goal_x, goal_y = goal_coordinate
        # Calculate the angle to the goal
        desired_angle = np.arctan2(
            goal_y - self.coordinates[1], goal_x - self.coordinates[0]
        )

        # Calculate the angle error
        angle_error = desired_angle - orientation

        # Normalize the angle error to the range [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        return angle_error

    def get_robot_state(self) -> list:
        """
        The state of the robot is the concatenation of the robot's velocity norm,
        the robot's orientation and the relative goal coordinates. Number of goals
        to consider is defined by the attribute nb_forseen_goal
        """
        current_visible_goal = self.get_current_visible_goal()
        if current_visible_goal is None or len(current_visible_goal) == 0:
            return [self.velocity_norm, self.theta, 0.0]

        # angle_error = self.get_angle_from_goal()
        relative_goal_coordinates = global_to_relative(
            np.array(current_visible_goal[0]),
            np.array(self.coordinates),
            self.orientation,
        )
        reaching_end_path = (
            len(relative_goal_coordinates.tolist()) < self.nb_forseen_goal
        )

        if reaching_end_path:
            # we add zeros to the relative goal coordinates to have a fixed size
            dummy_value = np.inf
            relative_goal_coordinates = np.concatenate(
                (
                    relative_goal_coordinates,
                    np.full(
                        (self.nb_forseen_goal - len(relative_goal_coordinates), 2),
                        dummy_value,
                    ),
                )
            )
        return np.concatenate(
            (self.velocity_norm, self.theta, relative_goal_coordinates), axis=None
        ).tolist()


def global_to_relative(global_coordinates, point_coordinates, point_orientation):
    # Translate the global coordinates so the point is at the origin
    translated_coordinates = global_coordinates - point_coordinates

    # Rotate the translated coordinates by the negative of the point's orientation
    rotation_matrix = np.array(
        [
            [np.cos(-point_orientation), -np.sin(-point_orientation)],
            [np.sin(-point_orientation), np.cos(-point_orientation)],
        ]
    )
    relative_coordinates = np.dot(rotation_matrix, translated_coordinates)

    return relative_coordinates
