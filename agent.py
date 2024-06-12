import numpy as np
import abc
import logging


class Agent(object):
    
    def __init__(self, visible:bool, desired_speed:float, radius:float, FOV:float, delta_t:float, **kwargs):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = visible
        self.desired_speed = desired_speed
        self.radius = radius
        self.FOV = np.pi * FOV
        self.coordinates = [None, None]
        self.speed = [None, None]
        self.orientation = None
        self.delta_t = delta_t
        self.arena_size = 6 if kwargs['arena_size'] is None else kwargs['arena_size']
        self.other_attribute = kwargs


        # TODO collect collection of goals from config file
        # TODO Move all that in Robot class 
        nb_goals = 5
        self.collection_goal_coordinates = np.random.uniform(-self.arena_size, self.arena_size, (nb_goals, 2))
        self.path = None
        self.current_goal_cusor = 0
        self.velocity_norm = 1.0
        self.acceleration_limits = [0.5, 2.0]
        self.robot_size = 0.3

    def __str__(self) -> str:
        return f"Agent: {self.__class__.__name__}"

    # def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
    #     self.px = px
    #     self.py = py
    #     self.gx = gx
    #     self.gy = gy
    #     self.vx = vx
    #     self.vy = vy
    #     self.theta = theta

    #     if radius is not None:
    #         self.radius = radius
    #     if v_pref is not None:
    #         self.v_pref = v_pref
        
    #     self.path = self.create_path()


    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    # def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
    #     self.px = px
    #     self.py = py
    #     self.gx = gx
    #     self.gy = gy
    #     self.vx = vx
    #     self.vy = vy
    #     self.theta = theta
    #     self.radius = radius
    #     self.v_pref = v_pref

        # self.path = self.create_path()

    # def get_observable_state(self):
    #     return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    # def get_observable_state_list(self):
    #     return [self.px, self.py, self.vx, self.vy, self.radius]

    # def get_observable_state_list_noV(self):
    #     return [self.px, self.py, self.radius]

    # def get_next_observable_state(self, action):
    #     self.check_validity(action)
    #     pos = self.compute_position(action, self.delta_t)
    #     next_px, next_py = pos
    #     if self.kinematics == 'holonomic':
    #         next_vx = action.vx
    #         next_vy = action.vy
    #     else:
    #         next_theta = self.theta + action.r
    #         next_vx = action.v * np.cos(next_theta)
    #         next_vy = action.v * np.sin(next_theta)
    #     return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    # def get_full_state(self):
    #     return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    # def get_full_state_list(self):
    #     return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    # def get_full_state_list_noV(self):
    #     return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta]
    #     # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref]

    def get_position(self) -> list:
        return self.coordinates

    def set_position(self, position: list) -> None:
        self.coordinates = position

    @abc.abstractmethod
    def get_goal_position(self):
        return
    
    # TODO move that to Robot class
    def get_current_goal(self):
        # TODO add to reach more than one goal in the future
        if self.current_goal_cusor >= len(self.collection_goal_coordinates):
            return None
        else:
            return self.collection_goal_coordinates[self.current_goal_cusor]

    def next_goal(self):   
        self.current_goal_cusor += 1
    
    def relative_state(self):
        robot_pos = self.get_position()
        current_goal = self.get_current_goal()
        if current_goal is None:
            return [0, 0, 0, 0]
        relative_goal = np.array(current_goal) - np.array(robot_pos)
        return [self.relative_speed ,self.theta ,relative_goal[0], relative_goal[1]]
    
    def get_agent_goal_collection(self):
        return self.collection_goal_coordinates


    def set_speed(self, speed: float):
        self.speed = speed


    @abc.abstractmethod
    def act(self, observation) -> list:
        """
        Compute state using received observation and pass it to policy

        """
        return

    # def check_validity(self, action):
    #     if self.kinematics == 'holonomic':
    #         assert isinstance(action, ActionXY)
    #     else:
    #         assert isinstance(action, ActionRot)

    # def compute_position(self, action, delta_t):
    #     self.check_validity(action)
    #     if self.kinematics == 'holonomic':
    #         px = self.px + action.vx * delta_t
    #         py = self.py + action.vy * delta_t
    #     # unicycle or bicycle
    #     elif self.kinematics == 'bicycle':
    #         self.theta += (action.v / self.robot_size) * np.tan(action.r) * delta_t
    #         self.px += action.v * np.cos(self.theta) * delta_t
    #         self.py += action.v * np.sin(self.theta) * delta_t   
    #         px = self.px
    #         py = self.py
    #     else:
    #         # naive dynamics
    #         # theta = self.theta + action.r * delta_t # if action.r is w
    #         # # theta = self.theta + action.r # if action.r is delta theta
    #         # px = self.px + np.cos(theta) * action.v * delta_t
    #         # py = self.py + np.sin(theta) * action.v * delta_t

    #         # differential drive
    #         epsilon = 0.0001
    #         if abs(action.r) < epsilon:
    #             R = 0
    #         else:
    #             w = action.r/delta_t # action.r is delta theta
    #             R = action.v/w

    #         px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r)
    #         py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)


    #     return px, py

    # def get_angle_from_goal(self):
    #     goal_coordinate = self.get_current_goal()
    #     orientation = self.theta % (2 * np.pi)
    #     goal_x, goal_y = goal_coordinate
    #     # Calculate the angle to the goal
    #     desired_angle = np.arctan2(goal_y - self.py, goal_x - self.px)

    #     # Calculate the angle error
    #     angle_error = desired_angle - orientation

    #     # Normalize the angle error to the range [-pi, pi]
    #     angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    #     return angle_error

    # def create_path(self):
    #     # the path is just a collection of line segments between the goals and the initial position
    #     initial_position_robot = self.get_position()
    #     goals_coordinates = self.get_agent_goal_collection()
    #     path_element = np.vstack((initial_position_robot, goals_coordinates))

    #     path = np.zeros((len(path_element)-1,4))

    #     for idx in range(len(path_element)-1):
    #         path[idx] = np.array([path_element[idx][0], path_element[idx][1], path_element[idx+1][0], path_element[idx+1][1]])
        
    #     # print(f"Path: {path}")
    #     # print(f"Goal: {goals_coordinates}")
    #     return path

    # def get_distance_from_path(self):
    #     position = self.get_position()
    #     idx = self.current_goal_cusor
    #     path = self.path[idx].reshape(2,2)

    #     if np.array_equal(position, path[0]):
    #         return 0.0

    #     a = position - path[0]
    #     b = path[1] - path[0]
    #     d = np.linalg.norm(a) * np.cos(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    #     normal_point = path[0] + d * b / np.linalg.norm(b)
    #     distance_to_path = np.linalg.norm(position - normal_point)
    #     # print(f"Distance to path: {distance_to_path}")
    #     return distance_to_path


    @abc.abstractmethod
    def step(self, action: list) -> None:
        """
        Perform an action and update the state
        """
        return
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        if self.kinematics == 'unicycle':
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
        if self.kinematics == 'bicycle':
            self.relative_speed = action.v

        pos = self.compute_position(action, self.delta_t)

        if self.kinematics == 'bicycle':
            self.vx = pos[0] - self.px
            self.vy = pos[1] - self.py
            
        self.px, self.py = pos

    # def one_step_lookahead(self, pos, action):
    #     px, py = pos
    #     # self.check_validity(action)
    #     new_px = px + action.vx * self.delta_t
    #     new_py = py + action.vy * self.delta_t
    #     new_vx = action.vx
    #     new_vy = action.vy
    #     return [new_px, new_py, new_vx, new_vy]

    # def reached_destination(self):
    #     return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    # def full_reset(self):
    #     nb_goals = 5
    #     arena_size = 6
    #     self.collection_goal_coordinates = np.random.uniform(-arena_size, arena_size, (nb_goals, 2))
    #     self.current_goal_cusor = 0
    #     self.path = self.create_path()
    
    # def get_future_traj(self, horizon):
    #     future_traj = []
    #     pos = self.get_position()
    #     for i in range(horizon):
    #         action = self.policy.act(None)
    #         pos = self.one_step_lookahead(pos, action)
    #         future_traj.append(pos)
    #     return future_traj


