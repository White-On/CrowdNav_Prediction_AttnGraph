from agent import Agent
import numpy as np


class Robot(Agent):
    def __init__(self, delta_t:float, nb_goals=5, nb_forseen_goal=2, is_visible=True, desired_speed=1.0, radius=0.2, FOV=np.pi, **kwargs):
        super().__init__(is_visible, desired_speed, radius, FOV, delta_t, **kwargs)

        self.nb_goals = nb_goals
        self.collection_goal_coordinates = [None]*self.nb_goals
        self.path = None
        self.current_goal_cusor = 0
        self.velocity_norm = None
        self.acceleration_limits = [0.5, 2.0]
        self.robot_size = 0.3
        self.orientation = 0
        self.theta = 0

    def __str__(self) -> str:
        return f'{super().__str__()}, {self.id = }'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def set_random_position(self) -> None:
        position_limit = self.arena_size 
        return [np.random.uniform(-position_limit, position_limit), 
                           np.random.uniform(-position_limit, position_limit)]
    
    def set_random_speed(self) -> None:
        return [np.random.uniform(-self.desired_speed, self.desired_speed), 
                      np.random.uniform(-self.desired_speed, self.desired_speed)]
    
    def set_random_goal(self) -> None:
        goal_coordinate_limit = self.arena_size 
        return [[np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit), 
                                            np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit)] for _ in range(self.nb_goals)]

    def create_path(self) -> list:
        # the path is just a collection of line segments between the goals and the initial position
        initial_position_robot = self.get_position()
        goals_coordinates = self.get_agent_goal_collection()
        no_initial_position = initial_position_robot == [None, None]
        no_goals = goals_coordinates == [None, None]
        if no_initial_position or no_goals:
            print(f'Warning: {self.__class__.__name__} has no initial position or goals set!')
            return None
        
        path_element = np.vstack((initial_position_robot, goals_coordinates))

        path = np.zeros((len(path_element)-1,4))

        for idx in range(len(path_element)-1):
            path[idx] = np.array([path_element[idx][0], path_element[idx][1], path_element[idx+1][0], path_element[idx+1][1]])
        
        # print(f"Path: {path}")
        # print(f"Goal: {goals_coordinates}")
        return path.tolist()
    
    def get_distance_from_path(self):
        position = self.get_position()
        idx = self.current_goal_cusor
        path = self.path[idx].reshape(2,2)

        if np.array_equal(position, path[0]):
            return 0.0

        a = position - path[0]
        b = path[1] - path[0]
        d = np.linalg.norm(a) * np.cos(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        normal_point = path[0] + d * b / np.linalg.norm(b)
        distance_to_path = np.linalg.norm(position - normal_point)
        # print(f"Distance to path: {distance_to_path}")
        return distance_to_path
    
    def reset(self) -> None:
        self.coordinates = self.set_random_position()
        self.speed = self.set_random_speed()
        self.goal_coordinates = self.set_random_goal()
        self.path = self.create_path()
        self.current_goal_cusor = 0
        self.velocity_norm = None
        self.orientation = np.random.uniform(0, 2*np.pi)

    # TODO EN BAS ICI
    
    def predict_what_to_do(self, *other_agent_state:list) -> list:
        return  
    
    def get_current_goal(self):
        # TODO add to reach more than one goal in the future
        if self.current_goal_cusor >= len(self.collection_goal_coordinates):
            return None
        else:
            return self.collection_goal_coordinates[self.current_goal_cusor]

    def next_goal(self):   
        self.current_goal_cusor += 1

    def is_goal_reached(self, threashold) -> bool:
        current_goal = self.get_current_goal()
        return np.linalg.norm(np.array(self.coordinates) - np.array(self.goal_coordinates)) < threashold
