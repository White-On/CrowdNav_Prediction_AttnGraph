import gym
import numpy as np

from crowd_sim.envs.crowd_sim_pred import CrowdSimPred
from crowd_sim.envs.utils.info import *
from numpy.linalg import norm
from crowd_sim.envs.utils.action import ActionRot, ActionXY

from rich import print


class CrowdSimCar(CrowdSimPred):
    '''
    Same as CrowdSimPred, except that
    The future human traj in 'spatial_edges' are dummy placeholders
    and will be replaced by the outputs of a real GST pred model in the wrapper function in vec_pretext_normalize.py
    '''
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super(CrowdSimCar, self).__init__()
        self.pred_method = None
        # to receive data from gst pred model
        self.gst_out_traj = None


    def set_robot(self, robot):
        """set observation space and action space"""
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        observation_space = {}
        forseen_index = 1
        # robot node: current speed, theta (wheel angle), objectives coordinates -> x and y coordinates * forseen_index
        vehicle_speed_boundries = [-0.5, 2]
        vehicle_angle_boundries = [-np.pi/6, np.pi/6]
        objectives_boundries = np.full((forseen_index * 2, 2), [-10,10])
        all_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries, objectives_boundries))
        observation_space['robot_node'] = gym.spaces.Box(low= all_boundries[:,0], high=all_boundries[:,1], dtype=np.float32)
        

        # # robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        # observation_space['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        observation_space['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        '''
        format of spatial_edges: [max_human_num, [state_t, state_(t+1), ..., state(t+self.pred_steps)]]
        '''
        # predictions only include mu_x, mu_y (or px, py)
        self.spatial_edge_dim = int(2*(self.predict_steps+1))

        observation_space['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.spatial_edge_dim),
                            dtype=np.float32)

        # masks for gst pred model
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        observation_space['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        observation_space['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        action_space_boundries = np.vstack((vehicle_speed_boundries, vehicle_angle_boundries))
        self.action_space = gym.spaces.Box(action_space_boundries[:,0], action_space_boundries[:,1], dtype=np.float32)

    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0


        self.humans = []
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))
        self.angle = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - self.robot.theta
        if self.angle > np.pi:
            # self.abs_angle = np.pi * 2 - self.abs_angle
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

        # get robot observation
        ob = self.generate_ob(reset=True, sort=self.config.args.sort_humans)

        return ob
    
    def step(self, action, update=True):
        """
        step function
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.robot.policy.name == 'ORCA':
            # assemble observation for orca: px, py, vx, vy, r
            # include all observable humans from t to t+t_pred
            _, _, human_visibility = self.get_num_human_in_fov()
            # [self.predict_steps + 1, self.human_num, 4]
            human_states = copy.deepcopy(self.calc_human_future_traj(method='truth'))
            # append the radius, convert it to [human_num*(self.predict_steps+1), 5] by treating each predicted pos as a new human
            human_states = np.concatenate((human_states.reshape((-1, 4)),
                                           np.tile(self.last_human_states[:, -1], self.predict_steps+1).reshape((-1, 1))),
                                          axis=1)
            # get orca action
            action = self.robot.act(human_states.tolist())
        else:
            # clip the action to be within the action space
            # action = self.robot.policy.clip_action(action, self.robot.v_pref)
            # print("action: ", action)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            # print("action clipped: ", action)
            if self.robot.kinematics == 'holonomic':
                action = ActionXY(action[0], action[1])
            else:
                action = ActionRot(action[0], action[1])

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
            action = ActionRot(self.desiredVelocity[0], action.r)

            # if action.r is delta theta
            action = ActionRot(self.desiredVelocity[0], action.r)
            if self.record:
                self.episodeRecoder.unsmoothed_actions.append(list(action))

            action = self.smooth_action(action)
        
        elif self.robot.kinematics == 'bicycle':
            pass
        

        human_actions = self.get_human_actions()

        # need to update self.human_future_traj in testing to calculate number of intrusions
        if self.phase == 'test':
            # use ground truth future positions of humans
            self.calc_human_future_traj(method='truth')

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action, danger_zone='future')


        if self.record:

            self.episodeRecoder.actionList.append(list(action))
            self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
            self.episodeRecoder.orientationList.append(self.robot.theta)

            if done:
                self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])
                self.episodeRecoder.saveEpisode(self.case_counter['test'])

        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter = self.step_counter+1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - 1
                else:
                    max_remove_num = (self.human_num - 1) - max(self.observed_human_ids)
                remove_num = np.random.randint(low=0, high=min(self.human_num_range, max_remove_num) + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)


        # compute the observation
        ob = self.generate_ob(reset=False)


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing and not self.record:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.humans[i] = self.generate_circle_crossing_human()
                    self.humans[i].id = i

        return ob, reward, done, info

    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recv from the env
        :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
        output predicted traj and output masks
        :return: True means received
        """
        self.gst_out_traj=data
        return True


    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=False):
        """Generate observation for reset and step functions"""
        # since gst pred model needs ID tracking, don't sort all humans
        # inherit from crowd_sim_lstm, not crowd_sim_pred to avoid computation of true pred!
        # sort=False because we will sort in wrapper in vec_pretext_normalize.py later
        ob = {}

        # nodes
        _, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        ob["robot_node"] = self.robot.relative_state()
        # print("robot_node: ", ob["robot_node"])

        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # ([relative px, relative py, disp_x, disp_y], human id)
        all_spatial_edges = np.ones((self.max_human_num, 2)) * np.inf

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                all_spatial_edges[self.humans[i].id, :2] = relative_pos

        ob['visible_masks'] = np.zeros(self.max_human_num, dtype=np.bool)
        # sort all humans by distance (invisible humans will be in the end automatically)
        if sort:
            ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x)))
            # after sorting, the visible humans must be in the front
            if num_visibles > 0:
                ob['visible_masks'][:num_visibles] = True
        else:
            ob['spatial_edges'] = all_spatial_edges
            ob['visible_masks'][:self.human_num] = self.human_visibility
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]

        parent_ob = ob

        ob = {}

        ob['visible_masks'] = parent_ob['visible_masks']
        ob['robot_node'] = parent_ob['robot_node']
        ob['temporal_edges'] = parent_ob['temporal_edges']

        ob['spatial_edges'] = np.tile(parent_ob['spatial_edges'], self.predict_steps+1)

        ob['detected_human_num'] = parent_ob['detected_human_num']

        return ob
    
    def compute_distance_from_human(self):
        distance_from_human = np.zeros(self.human_num)

        for i, human in enumerate(self.humans):
            distance_from_human[i] = np.linalg.norm(np.array([human.px, human.py]) - np.array([self.robot.px, self.robot.py]))

        return distance_from_human
    
    def compute_collision_reward(self, distance_from_human):
        distance_limit = 1

        collision_happed = np.min(distance_from_human) < distance_limit

        if collision_happed:
            return -40.0
        else:
            return 0.0
    
    def compute_near_collision_reward(self, distance_from_human):
        min_distance_to_keep_from_human = 1.5
        distance_to_closest_human = np.min(distance_from_human)
        # TODO get the corect velocity
        vehicle_current_speed = self.robot.relative_speed
        # TODO same here
        vehicle_min_acceleration = self.robot.acceleration_limits[0]

        dr = np.max([min_distance_to_keep_from_human, (vehicle_current_speed**2.0)/(2.0*vehicle_min_acceleration)])

        return np.exp((distance_to_closest_human-dr)/dr)

    def compute_speed_reward(self,current_speed, pref_speed):
        if 0.0 < current_speed <= pref_speed:
            # l = 1/pref_speed # old formula
            # return l * (pref_speed - current_speed)
            return 1-(pref_speed - current_speed)/pref_speed
        elif current_speed > pref_speed:
            return np.exp(-current_speed + pref_speed)
        elif current_speed <= 0.0:
            return current_speed
    
    def compute_angular_reward(self, angle):
        # TODO Careful with hard coded values
        angle_penalty = 20
        return np.exp(-angle/angle_penalty)

    def compute_proximity_reward(self, distance_from_goal):
        # TODO Careful with hard coded values
        penalty_distance = 2
        return 1 - 2 / (1 + np.exp(-distance_from_goal + penalty_distance))

    def calc_reward(self, action=None, danger_zone='future'):
        
        distance_from_human = self.compute_distance_from_human()

        collision_reward = self.compute_collision_reward(distance_from_human)
        near_collision_reward = self.compute_near_collision_reward(distance_from_human)
        speed_reward = self.compute_speed_reward(self.robot.relative_speed, self.robot.v_pref)
        angle_from_goal = np.abs(self.robot.get_angle_from_goal())
        # print(f'angle_from_goal: {np.degrees(angle_from_goal)}')
        angular_reward = self.compute_angular_reward(np.degrees(angle_from_goal))

        # TODO Je pense que ca ne vas pas fonctionner ici
        current_goal_coordinates = self.robot.get_current_goal()
        distance_from_goal = np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array(current_goal_coordinates))
        distance_from_path = self.robot.get_distance_from_path()
        proximity_reward = self.compute_proximity_reward(distance_from_path)

        reward = collision_reward + near_collision_reward + speed_reward + angular_reward + proximity_reward

        # print(f'💥collision_reward: {collision_reward:>7.2f}, 🚸 near_collision_reward: {near_collision_reward:>7.2f}, 🚀 speed_reward: {speed_reward:>7.2f}, 📐 angular_reward: {angular_reward:>7.2f}, 🤏 proximity_reward: {proximity_reward:>7.2f}, 🏆 reward: {reward:>7.2f}')

        episode_timeout = self.global_time >= self.time_limit - 1
        collision_happened = collision_reward < 0
        # TODO check if we are at the last goal and close enough to it
        goal_distance_threshold = 0.5
        reward_all_goals_reached = 50

        goal_reached = distance_from_goal <= goal_distance_threshold
        # print(f'🎯 distance_from_goal: {distance_from_goal:>7.2f}, 🎯 goal_distance_threshold: {goal_distance_threshold:>7.2f}, 🎯 goal_reached: {goal_reached:>7.2f}')

        if goal_reached:
            self.robot.next_goal()

            no_more_goal = self.robot.get_current_goal() is None
            if no_more_goal:
                goal_reached = True
                reward += reward_all_goals_reached
            else:
                goal_reached = False

        # TODO si le robot est asser proche du but on lui donnera et on passe au goal suivant ?

        conditions = {
            episode_timeout: Timeout,
            collision_happened: Collision,
            goal_reached: ReachGoal
        }

        for condition, result in conditions.items():
            if condition:
                done = True
                episode_info = result()
                break
        else:
            done = False
            episode_info = Nothing()

        return reward, done, episode_info


    def render(self, mode='human'):
        """
        render function
        use talk2env to plot the predicted future traj of humans
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

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
        artists=[]

        # add goal
        # goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        # ax.add_artist(goal)
        # artists.append(goal)

        # add list of goals
        for idx, goal in enumerate(self.robot.get_agent_goal_collection()):
            color = 'g'
            if idx == self.robot.goal_cusor:
                color = 'r'
            goal_point=mlines.Line2D([goal[0]], [goal[1]], color=color, marker='p', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal_point)
            artists.append(goal_point)
            # add a number to each goal
            offset_txt = 0.1
            plt.text(goal[0]-offset_txt, goal[1]-offset_txt, idx, color='black', fontsize=12)
        
        # add line for the path between goals
        if len(self.robot.path) > 0:
            for i in range(len(self.robot.path)):
                # print(f"Path: {self.robot.path[i]}")
                path_line = mlines.Line2D([self.robot.path[i][0], self.robot.path[i][2]], [self.robot.path[i][1], self.robot.path[i][3]], color='g', linestyle='--')
                ax.add_artist(path_line)
                artists.append(path_line)


        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' or self.robot.kinematics == 'bicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius+self.config.humans.radius, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5

        # plot the current human states
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.human_visibility[i]:
                human_circles[i].set_color(c='b')
            else:
                human_circles[i].set_color(c='r')

            if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
                i].py <= actual_arena_size:
                # label numbers on each human
                # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
                plt.text(self.humans[i].px , self.humans[i].py , i, color='black', fontsize=12)

        # plot predicted human positions
        for i in range(len(self.humans)):
            # add future predicted positions of each human
            if self.gst_out_traj is not None:
                for j in range(self.predict_steps):
                    circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
                                        self.config.humans.radius, fill=False, color='tab:orange', linewidth=1.5)
                    # circle = plt.Circle(np.array([robotX, robotY]),
                    #                     self.humans[i].radius, fill=False)
                    ax.add_artist(circle)
                    artists.append(circle)

        plt.pause(0.1)
        # plt.pause(1)
        # plt.pause(5)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
