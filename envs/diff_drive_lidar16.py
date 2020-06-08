'''
Author: Mihir Dharmadhikari
email: mihir.dharmadhikari@gmail.com

Openai gym environment for differential drive robot with 2D lidar
Number of lidar beams can be changed
All obstacles are circular and spawned randomly at each episode. Obstacles are not allowed to intersect, this is taken care during spawning
The goal point is always (0.0, 0.0)

Actions: linear velocity [0.0, 1.0], angular velocity [-2.0, 2.0]
Observations: lidar ranges(all beams), straight line distance to goal, Angular deviation between current heading and straight line towards goal
'''
import math
import gym
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# Required constants
kDg = 1.0  # Weight to dist
kDd = 0.5  # Weight to dir

MAX_STEPS = 1000
COLLISION_PENALTY = -5.0
ROBOT_RADIUS = 0.3
GOAL_THRESHOLD = ROBOT_RADIUS
LIDAR_BEAMS = 16
LIDAR_RANGE = 10.0
PI = math.pi
MIN_OBS_NUM = 15
MAX_OBS_NUM = 30
MIN_OBS_RAD = 0.5
MAX_OBS_RAD = 2.5
# Params for rendering
RENDER_SCALE = 50*2/3.0  # Experimentally found
RENDER_X_OFF = 500
RENDER_Y_OFF = 500
MAX_VEL = np.array([1.0, 0.5])
MIN_VEL = np.array([0.0, -0.5])
# All angles are in the range of -PI to PI
def correctAngle(angle):
	if angle > PI:
		angle = angle - 2*PI
	elif angle < -PI:
		angle = (2*PI + angle)
	return angle


class Ray(object):
	def __init__(self, angle):
		self.eqn = [0,-1,0]
		self.angle = angle  # Fixed angle w.r.t robot
		self.rng = np.inf
		self.end_pt = np.array([np.inf, np.inf])

	def setEqn(self, l):
		self.eqn = l

	def setRange(self, rng):
		self.rng = rng

	def reset(self):
		self.rng = np.inf


class DiffDriveLidar16(gym.Env):
	metadata = {
			'render.modes': ['human', 'rgb_array'],
			'video.frames_per_second' : 50
	}

	def __init__(self):
		# Play area
		self.area_min = np.array([-15.0, -15.0])
		self.area_max = np.array([15.0, 15.0])

		self.viewer = None

		# Setting action space
		# self.action_space = spaces.Box(low, high)
		# self.action_space = spaces.MultiDiscrete([3,3])
		self.action_space = spaces.Discrete(3)

		beams_max = np.ones(LIDAR_BEAMS)*np.inf
		beams_min = np.zeros(LIDAR_BEAMS)
		max_goal_x_dist = max(abs(self.area_min[0]), abs(self.area_max[0]))
		max_goal_y_dist = max(abs(self.area_min[1]), abs(self.area_max[1]))
		max_goal_dist = math.sqrt(max_goal_x_dist**2 + max_goal_y_dist**2)
		low = np.append(beams_min, np.array([0.0, -PI, -PI]))
		high = np.append(beams_max, np.array([max_goal_dist, PI, PI]))
		self.observation_space = spaces.Box(low, high)

		self.goal = np.array([0.0,0.0])
		self.curr_pos = np.array([0.0,0.0])
		self.curr_vel = np.array([0.0,0.0])
		self.curr_yaw = 0.0
		self.dt = 0.1
		self.prev_goal_dist = 0.0
		self.pre_goal_dir_error = 0.0
		self.prev_obs_dist = 0.0

		self.steps = 0

		self.obstacles = []  # List of [np.array([x,y]),radius]

		self.ranges = []  # This stores the distance at which a beam hit. Inf for those who did not hit
		self.rays = []  # List of Ray object
		for i in range(0,LIDAR_BEAMS):
			ray = Ray(-PI + i*PI/(LIDAR_BEAMS/2))
			self.rays.append(ray)
			self.ranges.append(np.inf)

	def step(self, action):
		for ray in self.rays:
			ray.reset()
		
		collision = 0.0
		oob = 0.0
		max_steps = 0.0
		goal_reached = 0.0
		
		# Rescalling from [0,1,2] to [-0.1,0,0.1]
		# delta_v = (action[0] - 1)*0.1
		# delta_omega = (action[1] - 1)*0.1
		delta_omega = (action - 1)*0.1

		# print(delta_v, delta_omega)

		self.curr_vel[1] = self.curr_vel[1] + delta_omega  # Angular velocity
		if self.curr_vel[1] > MAX_VEL[1]:
			self.curr_vel[1] = MAX_VEL[1]
		elif self.curr_vel[1] < MIN_VEL[1]:
			self.curr_vel[1] = MIN_VEL[1]

		# self.curr_vel[0] = self.curr_vel[0] + delta_v  # Linear velocity
		# if self.curr_vel[0] > MAX_VEL[0]:
		# 	self.curr_vel[0] = MAX_VEL[0]
		# elif self.curr_vel[0] < MIN_VEL[0]:
		# 	self.curr_vel[0] = MIN_VEL[0]
		
		self.curr_vel[0] = 0.5  # Fixed vel for now

		self.curr_yaw = self.curr_yaw + self.curr_vel[1]*self.dt
		if self.curr_yaw > PI:
			self.curr_yaw = (self.curr_yaw - 2*PI)
		elif self.curr_yaw < -PI:
			self.curr_yaw = (2*PI + self.curr_yaw)

		self.curr_pos[0] = self.curr_pos[0] + self.curr_vel[0]*self.dt*math.cos(self.curr_yaw)
		self.curr_pos[1] = self.curr_pos[1] + self.curr_vel[0]*self.dt*math.sin(self.curr_yaw)		

		
		delta_goal = self.goal - self.curr_pos
		goal_dist = np.linalg.norm(delta_goal)
		goal_dir_error = correctAngle(correctAngle(math.atan2(delta_goal[1], delta_goal[0])) - self.curr_yaw)
		# print(correctAngle(math.atan2(delta_goal[1], delta_goal[0])), self.curr_yaw, goal_dir_error)
		# delta_g = np.sign(self.prev_goal_dist - goal_dist)
		# print("Pos: current:", goal_dist, "prev:", self.prev_goal_dist)
		# print("Vel:", self.curr_vel)
		# if goal_dist < self.prev_goal_dist:
		# 	delta_g = (15.0*math.sqrt(2.0) - goal_dist)/(15.0*math.sqrt(2.0))
		# else:
		# 	delta_g = -(15.0*math.sqrt(2.0) - goal_dist)/(15.0*math.sqrt(2.0))
			
		# if goal_dist < self.prev_goal_dist:
		# 	delta_g = 1
		# else:
		# 	delta_g = 0
		delta_g = self.prev_goal_dist - goal_dist
		# print(delta_g)
			
		self.prev_goal_dist = goal_dist

		delta_dir = np.sign(abs(self.pre_goal_dir_error) - abs(goal_dir_error))
		self.pre_goal_dir_error = goal_dir_error

		# Max steps
		self.steps += 1
		if self.steps > MAX_STEPS:
			max_steps = 1.0

			
		# Out of bounds
		if(self.curr_pos[0] < self.area_min[0] or
		   self.curr_pos[1] < self.area_min[1] or
		   self.curr_pos[0] > self.area_max[0] or
		   self.curr_pos[1] > self.area_max[1]):
			oob = 1.0

		# Goal reached check
		if np.linalg.norm(self.goal - self.curr_pos) < GOAL_THRESHOLD:
			goal_reached = 1.0

		# Check collision
		obs_dists = []
		for obs in self.obstacles:
			delta = obs[0]-self.curr_pos
			obst_dist = np.linalg.norm(delta)
			obs_dists.append(obst_dist)
			if obst_dist < (obs[1] + ROBOT_RADIUS):
				collision = 1.0
				break

		# Ray casting
		self.rayCast()
		# 0.0 reward for going away from obstacle, -1.0 for going closer
		# Objective: not going close to obstacles, it is not to go away from obstacles
		front_obst_dist = self.ranges[int(LIDAR_BEAMS/2 - 1)]
		if front_obst_dist >= self.prev_obs_dist:
			delta_obs_rew = 0.0
		else:
			if front_obst_dist > 2.0*float(LIDAR_RANGE)/3.0:
				delta_obs_rew = 0.0
			else:
				delta_obs_rew = (-(2.0*float(LIDAR_RANGE)/3.0 - front_obst_dist))/(2.0*float(LIDAR_RANGE)/3.0)
			# delta_obs_rew = -1.0
		self.prev_obs_dist = front_obst_dist

		if abs(self.curr_vel[1]) > 0.0:
			delta_yaw_rew = -0.5
		else:
			delta_yaw_rew = 0.0

		# reward = (-collision) + (-oob) + goal_reached + delta_g + (-self.steps/MAX_STEPS)
		# if collision:
		# 	reward = COLLISION_PENALTY
		# elif max_steps:
		# 	reward = COLLISION_PENALTY
		# elif goal_reached:
		# 	reward = -2*COLLISION_PENALTY
		# else:
		# 	# print(delta_g, delta_obs_rew, 5*abs(delta_omega))
		# 	# print(delta_g, delta_obs_rew)
		# 	reward = delta_g - math.sqrt(delta_omega**2) - 0.1  # reward for going closer to goal, penalty for control action and each additional step
		# 	# reward = kDg * delta_g + kDd * delta_dir - self.steps/MAX_STEPS

		if goal_reached:
			reward = 1
		elif collision or oob:
			reward = -1
		else:
			reward = 10*delta_g - 0.01
			# print(reward)
		

		if collision or goal_reached or max_steps or oob:
			done = True
		else:
			done = False 

		info = {}
		if goal_reached:
			info['goal_reached'] = True
		else:
			info['goal_reached'] = False
		state = np.append(np.array(self.ranges), np.array([goal_dist, goal_dir_error, self.curr_yaw]))
		return state, reward, done, info


	def reset(self):
		self.steps = 0
		# Reset rendering
		if self.viewer is not None:
			self.viewer.close()	
			self.viewer = None
			

		# Set current pose
		# self.curr_pos = np.array([-10.0,1.0])
		self.curr_pos[0] = random.uniform(self.area_min[0], self.area_max[0])
		self.curr_pos[1] = random.uniform(self.area_min[1], self.area_max[1])
		self.curr_vel = np.array([0.0,0.0])
		self.curr_yaw = 0.0
		# self.curr_yaw = random.uniform(-PI, PI)

		# Set obstacles
		self.obstacles.clear()
		# Wall at periphery
		peripheri_res = 0.5
		pts_x = [self.area_min[0] + i*peripheri_res for i in range(0,int((self.area_max[0]-self.area_min[0])/peripheri_res))]
		for x in pts_x:
			self.obstacles.append([np.array([x,self.area_min[1]]), peripheri_res/2.0])
			self.obstacles.append([np.array([x,self.area_max[1]]), peripheri_res/2.0])
		pts_y = [self.area_min[1] + i*peripheri_res for i in range(0,int((self.area_max[1]-self.area_min[1])/peripheri_res))]
		for y in pts_y:
			self.obstacles.append([np.array([self.area_min[0], y]), peripheri_res/2.0])
			self.obstacles.append([np.array([self.area_max[0], y]), peripheri_res/2.0])
		# Random obstacles in the map
		obs_dists = []
		num_obs = random.randrange(MIN_OBS_NUM, MAX_OBS_NUM)
		# num_obs = 0
		for i in range(0,num_obs):
			obs_pos = np.zeros(2)
			obs_pos[0] = random.uniform(self.area_min[0], self.area_max[0])
			obs_pos[1] = random.uniform(self.area_min[1], self.area_max[1])
			obs_rad = random.uniform(MIN_OBS_RAD, MAX_OBS_RAD)
			resampling = False
			obst_dist = np.linalg.norm(obs_pos-self.curr_pos)
			if np.linalg.norm(obs_pos-self.goal) <= (obs_rad + 2*ROBOT_RADIUS) or obst_dist <= (obs_rad + 2*ROBOT_RADIUS):
				i -= 1
				continue
			for obs in self.obstacles:
				if np.linalg.norm(obs[0]-obs_pos) < (obs[1] + obs_rad):
					i -= 1
					resampling = True
					break
			if resampling:
				continue
			obs_dists.append(obst_dist)
			self.obstacles.append([obs_pos, obs_rad])
		
		self.rayCast()
		delta_goal = self.curr_pos - self.goal
		goal_dist = np.linalg.norm(delta_goal)
		goal_dir_error = correctAngle(correctAngle(math.atan2(delta_goal[1], delta_goal[0])) - self.curr_yaw)
		self.prev_goal_dist = goal_dist
		self.pre_goal_dir_error = goal_dir_error
		self.prev_obs_dist = self.ranges[int(LIDAR_BEAMS/2 - 1)]
		state = np.append(np.array(self.ranges), np.array([goal_dist, goal_dir_error, self.curr_yaw]))
		return state


	def render(self, mode='human', close=True):

		screen_width = 1000
		screen_height = 1000

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

			#Robot
			robot = self.viewer.draw_circle(RENDER_SCALE * ROBOT_RADIUS)
			robot.set_color(0.0,0.0,1.0)
			self.robottrans = rendering.Transform()
			robot.add_attr(self.robottrans)
			self.viewer.add_geom(robot)
			#Heading
			self.heading = self.viewer.draw_line((0.0, 0.0), (0.0, 0.0))
			self.viewer.add_geom(self.heading)
			
			#Obstacles
			for obs in self.obstacles:
				obs1 = self.viewer.draw_circle(RENDER_SCALE*obs[1])
				obs1_trans = rendering.Transform()
				obsx = RENDER_SCALE*obs[0][0] + RENDER_X_OFF
				obsy = RENDER_SCALE*obs[0][1] + RENDER_Y_OFF
				obs1_trans.set_translation(obsx, obsy)
				obs1.add_attr(obs1_trans)
				obs1.set_color(1.0,0.1,0.1)
				self.viewer.add_geom(obs1)
			
			# Goal
			goal_obj = self.viewer.draw_circle(RENDER_SCALE*2*ROBOT_RADIUS)
			goal_trans = rendering.Transform()
			goal_rt = self.rT(self.goal)
			goal_trans.set_translation(goal_rt[0], goal_rt[1])
			goal_obj.add_attr(goal_trans)
			goal_obj.set_color(0.0,1.0,0.0) #Green
			self.viewer.add_geom(goal_obj)

			# Rays
			self.render_rays = []
			for ray in self.rays:
				render_ray = self.viewer.draw_line((self.rT(self.curr_pos)[0], self.rT(self.curr_pos)[1]), (self.rT(ray.end_pt)[0], self.rT(ray.end_pt)[1]))
				self.render_rays.append(render_ray)
				
				render_ray.linewidth.stroke = 2
				render_ray.set_color(1.0, 0.3, 0.0)
				self.viewer.add_geom(render_ray)

		xrobot = RENDER_SCALE*self.curr_pos[0] + RENDER_X_OFF
		yrobot = RENDER_SCALE*self.curr_pos[1] + RENDER_Y_OFF
		self.robottrans.set_translation(xrobot, yrobot)
		self.robottrans.set_rotation(self.curr_yaw)

		heading_start_x = RENDER_SCALE*self.curr_pos[0] + RENDER_X_OFF
		heading_start_y = RENDER_SCALE*self.curr_pos[1] + RENDER_Y_OFF
		heading_end_x = RENDER_SCALE*(self.curr_pos[0] + 3*ROBOT_RADIUS*math.cos(self.curr_yaw)) + RENDER_X_OFF
		heading_end_y = RENDER_SCALE*(self.curr_pos[1] + 3*ROBOT_RADIUS*math.sin(self.curr_yaw)) + RENDER_Y_OFF
		self.heading.start = (heading_start_x, heading_start_y)
		self.heading.end = (heading_end_x, heading_end_y)
		self.heading.set_color(0.0,0.5,0.5)
		self.heading.linewidth.stroke = 5

		for i in range(len(self.render_rays)):
			self.render_rays[i].start = (self.rT(self.curr_pos)[0], self.rT(self.curr_pos)[1])
			self.render_rays[i].end = (self.rT(self.rays[i].end_pt)[0], self.rT(self.rays[i].end_pt)[1])
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	def rT(self, a):
		return np.array([RENDER_SCALE*a[0] + RENDER_X_OFF, RENDER_SCALE*a[1] + RENDER_Y_OFF])

	def rayCast(self):
		obstacles_detailed = []  # Obstacles with distance from robot
		# Set ray equationsles_detailed = []  # Obstacles with distance from robot
		for obs in self.obstacles:
			delta = obs[0]-self.curr_pos
			obst_dist = np.linalg.norm(delta)

			obst_yaw = math.atan2(delta[1], delta[0])
			obstacles_detailed.append(obs+[obst_dist, obst_yaw])
			
		obstacles_detailed.sort(key=lambda obs:obs[2])
		# Set ray equations
		for i in range(0,LIDAR_BEAMS):
			m = math.tan(correctAngle(self.curr_yaw + self.rays[i].angle))
			l = []
			l.append(m)
			l.append(-1)
			l.append(self.curr_pos[1] - m*self.curr_pos[0])
			self.rays[i].setEqn(l)
		# Do ray casting
		for obs in obstacles_detailed:
			for ray in self.rays:
				if not np.isinf(ray.rng):  # This ray has already hit an obstacle
					continue
				l = ray.eqn
				# Check if this obstacle is in the range of this ray
				if abs(correctAngle(ray.angle + self.curr_yaw) - obs[3]) < PI/(LIDAR_BEAMS/2) or abs(correctAngle(ray.angle + self.curr_yaw) - obs[3]) > 2*PI - PI/(LIDAR_BEAMS/2):
					d = abs((l[0]*obs[0][0] + l[1]*obs[0][1] + l[2]) / 
										math.sqrt(l[0]**2 + l[1]**2))
					if d > obs[1]:
						continue
					ray_dist = (math.sqrt((self.curr_pos[0]-obs[0][0])**2 + (self.curr_pos[1]-obs[0][1])**2 - d**2) - 
											math.sqrt(obs[1]**2 - d**2))
					ray.setRange(ray_dist)

		# Calucate the end point coordinates of the ray
		for i in range(0,len(self.ranges)):
			if self.rays[i].rng > LIDAR_RANGE:
				self.rays[i].rng = LIDAR_RANGE
			self.ranges[i] = self.rays[i].rng
			# If the ray has hit something set the end point at that point
			# Else set it to the periphery of the lidar range
			
			if not np.isinf(self.rays[i].rng):
				end_pt_x = self.curr_pos[0] + self.rays[i].rng*math.cos(correctAngle(self.rays[i].angle + self.curr_yaw))
				end_pt_y = self.curr_pos[1] + self.rays[i].rng*math.sin(correctAngle(self.rays[i].angle + self.curr_yaw))
				self.rays[i].end_pt = np.array([end_pt_x, end_pt_y])
			else:
				end_pt_x = self.curr_pos[0] + LIDAR_RANGE*math.cos(correctAngle(self.rays[i].angle + self.curr_yaw))
				end_pt_y = self.curr_pos[1] + LIDAR_RANGE*math.sin(correctAngle(self.rays[i].angle + self.curr_yaw))
				self.rays[i].end_pt = np.array([end_pt_x, end_pt_y])
