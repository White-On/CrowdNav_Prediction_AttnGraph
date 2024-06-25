from .agent import Agent
import numpy as np
import rvo2


class Human(Agent):
    HUMAN_LIST = []

    def __init__(
        self,
        delta_t: float,
        is_moving=True,
        is_visible=True,
        desired_speed=1.0,
        radius=0.2,
        FOV=np.pi,
        **kwargs,
    ):
        super().__init__(is_visible, desired_speed, radius, FOV, delta_t, **kwargs)
        Human.HUMAN_LIST.append(self)

        self.is_moving = is_moving
        self.goal_coordinates = [None, None]

    @classmethod
    def apply(cls, function: staticmethod, *args, **kwargs) -> list:
        return [function(human, *args, **kwargs) for human in cls.HUMAN_LIST]

    def __str__(self) -> str:
        return f"{super().__str__()}, {self.id = }"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, action: list) -> None:
        if self.is_moving:
            self.coordinates = self.compute_position(action)
            self.speed = action

    def compute_position(self, action: list) -> list:
        computed_position = self.coordinates + np.array(action) * self.delta_t
        return computed_position.tolist()

    def set_random_goal(self) -> list:
        goal_coordinate_limit = self.arena_size
        return [
            np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit),
            np.random.uniform(-goal_coordinate_limit, goal_coordinate_limit),
        ]

    def predict_what_to_do(self, *other_agent_state: list) -> list:
        """
        other_agent_state: list of other agents' states in the form of [x, y, vx, vy]
        We run a RVO2 simulation to predict the next action to take.
        TODO : maybe use something else to predict the next action to take
        beceause creating a new simulation for each agent is not efficient.
        """
        no_goal_set = self.goal_coordinates == [None, None]
        if no_goal_set:
            self.goal_coordinates = self.set_random_goal()

        max_neighbors = len(other_agent_state)
        timehorizon = 5
        time_horizon_obst = 5
        neighbor_dist = 20
        safe_distance = 0.15

        # just for redability
        params = [neighbor_dist, max_neighbors, timehorizon, time_horizon_obst]

        speed_factor = 1
        goal_coordinates = np.array(self.goal_coordinates)
        human_coordinates = np.array(self.coordinates)
        velocity_toward_goal = goal_coordinates - human_coordinates
        normalized_velocity = velocity_toward_goal / np.linalg.norm(
            velocity_toward_goal
        )
        speed_to_take = speed_factor * normalized_velocity

        no_neighbors = max_neighbors == 0
        if no_neighbors:
            return speed_to_take.tolist()

        simulation = rvo2.PyRVOSimulator(
            self.delta_t, *params, self.radius, self.desired_speed
        )
        # add self to simulation
        simulation.addAgent(
            (self.coordinates[0], self.coordinates[1]),
            *params,
            self.radius + safe_distance,
            self.desired_speed,
            (self.speed[0], self.speed[1]),
        )

        for i, state in enumerate(other_agent_state):
            simulation.addAgent(
                (state[0], state[1]),
                *params,
                self.radius + safe_distance,
                self.desired_speed,
                (state[2], state[3]),
            )

        # Set the preferred velocity to be a vector of magnitude x in the direction of the goal.
        simulation.setAgentPrefVelocity(0, tuple(speed_to_take))
        for i in range(len(other_agent_state)):
            simulation.setAgentPrefVelocity(i + 1, (0, 0))

        simulation.doStep()
        return list(simulation.getAgentVelocity(0))

    def is_goal_reached(self, threashold) -> bool:
        return (
            np.linalg.norm(np.array(self.coordinates) - np.array(self.goal_coordinates))
            < threashold
        )

    def reset(self) -> None:
        self.coordinates = self.set_random_position(position_limit=self.arena_size)
        self.speed = self.set_random_speed(speed_limit=self.desired_speed)
        self.goal_coordinates = self.set_random_goal()
