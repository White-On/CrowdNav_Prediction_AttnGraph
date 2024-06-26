import numpy as np
import abc
import logging


class Agent(object):
    ID_COUNTER = 0
    ENTITIES = []

    def __init__(
        self,
        is_visible: bool,
        desired_speed: float,
        radius: float,
        FOV: float,
        delta_t: float,
        **kwargs,
    ):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.is_visible = is_visible
        self.desired_speed = desired_speed
        self.radius = radius
        self.FOV = np.pi * FOV
        self.coordinates = [None, None]
        self.speed = [0, 0]
        self.orientation = 0
        self.delta_t = delta_t
        self.arena_size = kwargs.get("arena_size", 5)
        self.other_attribute = kwargs
        self.sensor_range = kwargs.get("sensor_range", 10)

        self.id = Agent.ID_COUNTER
        Agent.ID_COUNTER += 1
        Agent.ENTITIES.append(self)

    def __str__(self) -> str:
        return f"Agent: {self.__class__.__name__}"

    @classmethod
    def apply(cls, function: staticmethod, *args, **kwargs) -> list:
        return [function(agent, *args, **kwargs) for agent in cls.ENTITIES]

    def who_can_i_see(self, other_agent_positions: list) -> list:
        other_agent_positions = np.array(other_agent_positions)
        distance = np.linalg.norm(other_agent_positions - self.coordinates, axis=1)
        return distance < self.sensor_range

    def distance_from_other_agents(self, other_agent_positions: list) -> list:
        other_agent_positions = np.array(other_agent_positions)
        distance = np.linalg.norm(other_agent_positions - self.coordinates, axis=1)
        return distance.tolist()

    def can_i_see(self, other_agent: "Agent") -> bool:
        diff = np.array(other_agent.coordinates) - np.array(self.coordinates)
        distance_squared = np.dot(diff, diff)
        return distance_squared < self.sensor_range**2

    def get_position(self) -> list:
        return self.coordinates

    def set_position(self, position: list) -> None:
        self.coordinates = position

    @abc.abstractmethod
    def get_goal_position(self):
        return

    @abc.abstractmethod
    def reset(self) -> None:
        return

    @abc.abstractmethod
    def predict_what_to_do(self, *observations) -> list:
        """
        Compute state using received observation and pass it to policy

        """
        return

    @abc.abstractmethod
    def step(self, action: list) -> None:
        """
        Perform an action and update the state
        """
        return

    def set_random_position(self, position_limit: float) -> None:
        return [
            np.random.uniform(-position_limit, position_limit),
            np.random.uniform(-position_limit, position_limit),
        ]

    def set_random_speed(self, speed_limit) -> None:
        return [
            np.random.uniform(-speed_limit, speed_limit),
            np.random.uniform(-speed_limit, speed_limit),
        ]


class AgentGroup:
    def __init__(self, *agents):
        self.agents = agents

    def filter(self, condition) -> "AgentGroup":
        return AgentGroup(*[agent for agent in self.agents if condition(agent)])

    def apply(self, function: staticmethod, *args, **kwargs) -> list:
        return [function(agent, *args, **kwargs) for agent in self.agents]

    def get_all(self) -> list:
        return self.agents

    def __len__(self):
        return len(self.agents)

    def reset(self):
        return self.apply(lambda x: x.reset())

    def __repr__(self) -> str:
        return f"{self.agents}"
