import gym
import gym.spaces
import numpy as np

from human import Human
from robot import Robot
from agent import Agent, AgentGroup
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from rich import print

class CrowdSimCar(gym.Env):
    '''
    Same as CrowdSimPred, except that
    The future human traj in 'spatial_edges' are dummy placeholders
    and will be replaced by the outputs of a real GST pred model in the wrapper function in vec_pretext_normalize.py
    '''
    metadata = {"render_modes": ["human","debug", None]}
                
    def __init__(self, render_mode=None, arena_size=6, nb_pedestrians=10, episode_time=100, time_step=0.1, display_future_trajectory=True):
        self.arena_size = arena_size
        if render_mode not in self.metadata['render_modes']:
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
        self.goal_threshold_distance = 0.1

        sensor_range = 4
        self.robot = Robot(self.time_step, arena_size=arena_size, sensor_range=sensor_range,nb_forseen_goal=self.nb_time_steps_seen_as_graph_feature)
        for _ in range(nb_pedestrians):
            Human(self.time_step, arena_size=arena_size, sensor_range=sensor_range)

        self.observation_space = self.define_observations_space(self.robot.nb_forseen_goal, 
                                                                nb_humans=nb_pedestrians, 
                                                                nb_graph_feature=self.nb_time_steps_seen_as_graph_feature)
        self.action_space = self.define_action_space()
        self.all_agent_group = AgentGroup(*Agent.ENTITIES)


    def define_observations_space(self, forseen_index:int, nb_humans:int,nb_graph_feature:int)->gym.spaces.Dict:
        observation_space = {}
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        vehicle_speed_boundries = [-0.5, 2]
        vehicle_angle_boundries = [-np.pi/6, np.pi/6]
        objectives_boundries = np.full((forseen_index, 2), [-10,10])
        all_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries, objectives_boundries))
        observation_space['robot_node'] = gym.spaces.Box(low= all_boundries[:,0], high=all_boundries[:,1], dtype=np.float32)
        
        # predictions only include mu_x, mu_y (or px, py)
        spatial_edge_dim = int(2*(nb_graph_feature))

        observation_space['graph_features'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(nb_humans, spatial_edge_dim), dtype=np.float32)
        
        observation_space['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(nb_humans,),
                                            dtype=np.bool8)
            
        return gym.spaces.Dict(observation_space)
    
    def define_action_space(self)->gym.spaces.Box:
        # vehicle_speed_boundries = [-0.5, 2]
        # TODO put back the ability to drive backward
        vehicle_speed_boundries = [0.0, 2]
        limit_angle = np.deg2rad(14)
        # limit_angle = np.pi/6
        vehicle_angle_boundries = [-limit_angle, limit_angle]

        action_space_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries))
        return gym.spaces.Box(action_space_boundries[:,0], action_space_boundries[:,1], dtype=np.float32)

    def reset(self)->dict:
        """
        Reset the environment
        :return:
        """

        self.global_time = 0
        self.all_agent_group.reset()
        # get robot observation
        observation_after_reset = self.generate_observation()
        if self.render_mode is not None:
            self._render_frame(mode = self.render_mode)

        return observation_after_reset
    
    def step(self, robot_action:np.ndarray)->tuple:
        """
        step function
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        robot_action = np.clip(robot_action, self.action_space.low, self.action_space.high)
        
        # compute reward and episode info
        reward, done, episode_info = self.calc_reward()

        # apply action and update all agents
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)

        # Human step
        for human in Human.HUMAN_LIST:            
            other_agent_state = (agent_visible
                                .filter(lambda x: x.id != human.id)
                                .filter(human.can_i_see)
                                .apply(lambda x: x.coordinates + x.speed))
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

        self.global_time += self.time_step

        # compute the observation
        step_observation = self.generate_observation()

        if self.render_mode is not None:
            self._render_frame(mode = self.render_mode)

        return step_observation, reward, done, episode_info

    # def talk2Env(self, data):
    #     """
    #     Call this function when you want extra information to send to/recv from the env
    #     :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
    #     output predicted traj and output masks
    #     :return: True means received
    #     """
    #     self.gst_out_traj=data
    #     return True


    def generate_observation(self)->dict:
        """Generate observation for reset and step functions"""

        observation = {}
        nb_humans_in_simulation = len(Human.HUMAN_LIST)
        agent_visible = self.all_agent_group.filter(lambda x: x.is_visible)
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        observation['robot_node'] = np.array(self.robot.get_robot_state())
        
        # graph features: future position of every human + robot 
        # dim = [num_visible_humans + 1, 2*(self.predict_steps+1)]
        observation['graph_features'] = np.zeros((nb_humans_in_simulation, (self.nb_time_steps_seen_as_graph_feature), 2))
        # logging.info(f"graph_features: {observation['graph_features'].shape}")

        visible_agent_by_robot = (agent_visible
                                 .filter(lambda x: x.id != self.robot.id)
                                 .filter(self.robot.can_i_see))

        for i, human in enumerate(Human.HUMAN_LIST):            
            if human.id in visible_agent_by_robot.apply(lambda x: x.id):
                direction_vector = np.array(human.speed)
                direction_vector *= self.time_step
                # with the vector we calculate the n future positions
                human_position = np.array(human.get_position())
                direction_vector = np.tile(direction_vector, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 2) * np.arange(0, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 1)
                human_future_traj = human_position + direction_vector
            else:
                # the visibility mask will make sure that the invisible humans are not considered
                human_future_traj = np.zeros((self.nb_time_steps_seen_as_graph_feature, 2))
            observation['graph_features'][i] = human_future_traj
        
        # transform the graph features into relative coordinates
        # TODO/WARNING: Giving the robot relative position to the robot itself
        # does not make sense
        robot_position = np.array(self.robot.get_position())
        # add robot future traj
        # robot_future_traj = np.tile(self.robot.speed, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 2) * np.arange(0, self.nb_time_steps_seen_as_graph_feature).reshape(-1, 1)
        # observation['graph_features'][-1] = robot_future_traj
        observation['graph_features'] = observation['graph_features'] - robot_position
        observation['graph_features'] = observation['graph_features'].reshape(nb_humans_in_simulation, -1)

        list_of_visible_humans = visible_agent_by_robot.apply(lambda x: x.id)
        visibility_mask = [True if human.id in list_of_visible_humans else False for human in Human.HUMAN_LIST]
        observation['visible_masks'] = visibility_mask

        return observation
    
    
    def compute_collision_reward(self, distance_from_human:float)->float:
        distance_limit = self.robot.radius + Human.HUMAN_LIST[0].radius

        collision_happed = np.min(distance_from_human) < distance_limit

        if collision_happed:
            return -40.0
        else:
            return 0.0
    
    def compute_near_collision_reward(self, distance_from_human:float)->float:
        min_distance_to_keep_from_human = 1.5
        distance_to_closest_human = np.min(distance_from_human)
        vehicle_current_speed = self.robot.velocity_norm
        vehicle_min_acceleration = self.robot.acceleration_limits[0]

        dr = np.max([min_distance_to_keep_from_human, (vehicle_current_speed**2.0)/(2.0*vehicle_min_acceleration)])

        return np.exp((distance_to_closest_human-dr)/dr)

    def compute_speed_reward(self,current_speed:float, pref_speed:float)->float:
        if 0.0 < current_speed <= pref_speed:
            # l = 1/pref_speed # old formula
            # return l * (pref_speed - current_speed)
            return 1-(pref_speed - current_speed)/pref_speed
        elif current_speed > pref_speed:
            return np.exp(-current_speed + pref_speed)
        elif current_speed <= 0.0:
            return current_speed
    
    def compute_angular_reward(self, angle:float)->float:
        # TODO Careful with hard coded values
        angle_penalty = 20
        return np.exp(-angle/angle_penalty)

    def compute_proximity_reward(self, distance_from_goal:float)->float:
        # TODO Careful with hard coded values
        penalty_distance = 2
        return 1 - 2 / (1 + np.exp(-distance_from_goal + penalty_distance))

    def calc_reward(self)->tuple:
        
        distance_from_human = self.robot.distance_from_other_agents([human.get_position() for human in Human.HUMAN_LIST])

        collision_reward = self.compute_collision_reward(distance_from_human)
        near_collision_reward = self.compute_near_collision_reward(distance_from_human)
        speed_reward = self.compute_speed_reward(self.robot.velocity_norm, self.robot.desired_speed)
        angle_from_goal = np.abs(self.robot.get_angle_from_goal())
        # print(f'angle_from_goal: {np.degrees(angle_from_goal)}')
        angular_reward = self.compute_angular_reward(np.degrees(angle_from_goal))

        # TODO Je pense que ca ne vas pas fonctionner ici
        current_goal_coordinates = self.robot.get_current_visible_goal()[0]
        distance_from_goal = np.linalg.norm(np.array(self.robot.coordinates) - np.array(current_goal_coordinates))
        distance_from_path = self.robot.get_distance_from_path()
        proximity_reward = self.compute_proximity_reward(distance_from_path)

        reward = collision_reward + near_collision_reward + speed_reward + angular_reward + proximity_reward

        # print(f'💥collision_reward: {collision_reward:>7.2f}, 🚸 near_collision_reward: {near_collision_reward:>7.2f}, 🚀 speed_reward: {speed_reward:>7.2f}, 📐 angular_reward: {angular_reward:>7.2f}, 🤏 proximity_reward: {proximity_reward:>7.2f}, 🏆 reward: {reward:>7.2f}')

        episode_timeout = self.global_time >= self.episode_time - 1
        collision_happened = collision_reward < 0
        # TODO check if we are at the last goal and close enough to it
        reward_all_goals_reached = 50
        reward_single_goal_reached = 10

        # print(f'🎯 distance_from_goal: {distance_from_goal:>7.2f}, 🎯 goal_distance_threshold: {goal_distance_threshold:>7.2f}, 🎯 goal_reached: {goal_reached:>7.2f}')
        is_robot_reach_goal = self.robot.is_goal_reached(self.goal_threshold_distance)
        if is_robot_reach_goal:
            reward += reward_single_goal_reached
            self.robot.next_goal()
        all_goals_reached = self.robot.current_goal_cusor >= len(self.robot.collection_goal_coordinates)
        if all_goals_reached:
            logging.info('All robot goals are reached!')
            # self.all_agent_group.reset()
            reward += reward_all_goals_reached


        # TODO si le robot est asser proche du but on lui donnera et on passe au goal suivant ?

        conditions = {
            episode_timeout: 'Timeout',
            collision_happened: 'Collision',
            all_goals_reached: 'ReachGoal'
        }

        for condition, result in conditions.items():
            if condition:
                done = True
                episode_info = result
                break
        else:
            done = False
            episode_info = 'Nothing'

        return reward, done, episode_info


    def _render_frame(self, mode='human')->None:
        """
        render function
        """
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        human_color = 'gray'
        visible_human_color = 'blue'
        robot_goal_color = 'green'
        human_goal_color = 'lightblue'
        direction_arrow_color = 'black'
        color_robot_path = robot_goal_color
        robot_visual_radius = 10
        human_visual_radius = 6

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax=self.render_axis
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
                color = 'r'
            ax.plot(goal[0], goal[1], color=color, marker='p', markersize=markersize_goal, label='Goal')
            offset_txt = 0.05
            plt.text(goal[0]-offset_txt, goal[1]-offset_txt, idx, color='black', fontsize=markersize_goal-1)
        
        # add line for the path between goals
        if len(self.robot.path) > 0:
            for path in self.robot.path:
                ax.plot([path[0], path[2]], [path[1], path[3]], color=color_robot_path, linestyle='--')


        # add robot
        # ax.plot(*self.robot.get_position(), color=robot_color, marker='o', markersize=robot_visual_radius, label='Robot')
        ax.add_artist(patches.Rectangle((robotX - robot_radius, robotY - robot_radius), 3 * robot_radius, 2*robot_radius, color=robot_color, linewidth=0.5, angle=np.degrees(self.robot.orientation), rotation_point=(robotX, robotY)))
        # direction and goal arrow
        if mode == 'debug':
            ax.arrow(x=self.robot.coordinates[0], y=self.robot.coordinates[1], dx=self.robot.speed[0], dy=self.robot.speed[1], head_width=0.1, head_length=0.1, fc=direction_arrow_color, ec=direction_arrow_color)
            velocity_toward_goal = np.array(self.robot.get_current_visible_goal()[0]) - np.array(self.robot.coordinates)
            normalized_velocity = velocity_toward_goal / np.linalg.norm(velocity_toward_goal)
            ax.arrow(x=self.robot.coordinates[0], y=self.robot.coordinates[1], dx=normalized_velocity[0], dy=normalized_velocity[1], head_width=0.1, head_length=0.1, fc=robot_goal_color, ec=robot_goal_color)
        
        # draw FOV for the robot
        # TODO if the robot has a FOV diff than 360

        # add an arc of robot's sensor range
        ax.add_artist(plt.Circle((robotX, robotY), self.robot.sensor_range, fill=False, color='black', linestyle='--', linewidth=0.5))

        id_visible_agent_by_robot = (self.all_agent_group.filter(lambda x: x.id != self.robot.id)
                                    .filter(self.robot.can_i_see)
                                    .apply(lambda x: x.id))
        # add human
        for human in Human.HUMAN_LIST:
            color = human_color
            if human.id in id_visible_agent_by_robot:
                color = visible_human_color
            ax.plot(*human.get_position(), color=color, marker='o', markersize=human_visual_radius, label='Human')
            if mode == 'debug':
                ax.plot(*human.goal_coordinates, color=human_goal_color, marker='o', label='Human Goal')
                ax.arrow(x=human.coordinates[0], y=human.coordinates[1], dx=human.speed[0], dy=human.speed[1], head_width=0.1, head_length=0.1, fc=direction_arrow_color, ec=direction_arrow_color)
                velocity_toward_goal = np.array(human.goal_coordinates) - np.array(human.coordinates)
                normalized_velocity = velocity_toward_goal / np.linalg.norm(velocity_toward_goal)
                ax.arrow(x=human.coordinates[0], y=human.coordinates[1], dx=normalized_velocity[0], dy=normalized_velocity[1], head_width=0.1, head_length=0.1, fc=human_goal_color, ec=human_goal_color)

        if self.display_future_trajectory:
            observation = self.generate_observation()
            predicted_positions = observation['graph_features']
            visible_masks = observation['visible_masks']
            # we remove the predicted positions with the visibility mask
            predicted_positions = predicted_positions[visible_masks]
            for human_future_traj in predicted_positions:
                human_future_traj = human_future_traj.reshape(-1, 2)
                human_future_traj = human_future_traj + np.array([robotX, robotY])
                ax.plot(human_future_traj[:, 0], human_future_traj[:, 1], color='tab:orange', marker='o', markersize=human_visual_radius, label='Human Future Traj', alpha=0.5, linestyle='--')
        
        # # plot the current human states
        # for i in range(len(self.humans)):
        #     ax.add_artist(human_circles[i])
        #     artists.append(human_circles[i])

        #     # green: visible; red: invisible
        #     if self.human_visibility[i]:
        #         human_circles[i].set_color(c='b')
        #     else:
        #         human_circles[i].set_color(c='r')

        #     if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
        #         i].py <= actual_arena_size:
        #         # label numbers on each human
        #         # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
        #         plt.text(self.humans[i].px , self.humans[i].py , i, color='black', fontsize=12)

        # # plot predicted human positions
        # if self.gst_out_traj is not None:
        #     for i in range(len(self.humans)):
        #     # add future predicted positions of each human
        #         if self.human_visibility[i]:
        #             for j in range(self.predict_steps):
        #                 circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
        #                                     self.config.humans.radius, fill=False, color='tab:orange', linewidth=1.5, alpha=0.5)
        #                 # circle = plt.Circle(np.array([robotX, robotY]),
        #                 #                     self.humans[i].radius, fill=False)
        #                 ax.add_artist(circle)
        #                 artists.append(circle)
        if self.render_delay:
            plt.pause(self.render_delay)
        else:
            plt.pause(0.0001)

    def render(self, mode='human'):
        self._render_frame(mode)