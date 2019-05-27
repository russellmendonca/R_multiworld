from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
	create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv
from pyquaternion import Quaternion

def zangle_to_quat(zangle):
	"""
	:param zangle in rad
	:return: quaternion
	"""
	#return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, -1], angle= zangle)).elements
	return (Quaternion(axis=[0,0,1], angle=np.pi) * Quaternion(axis=[-1, 0, 0], angle= zangle)).elements
	#return (Quaternion(axis=[1,0,0], angle=np.pi) * Quaternion(axis=[0, -1, 0], angle= zangle)).elements
	#return (Quaternion(axis=[1,0,0], angle=np.pi) * Quaternion(axis=[-1, 0, 0], angle= zangle)).elements #fail
	#return (Quaternion(axis=[0,0,1], angle=np.pi) * Quaternion(axis=[0, -1, 0], angle= zangle)).elements
class SawyerCoffeeEnv( SawyerPushEnv):
	def __init__(
			self,
			tasks = [{'goal': np.array([0, 1.0, 0.05]), 'height': 0.06, 'obj_init_pos':np.array([0, 0.6, 0.04])}] , 
			hand_type = 'weiss_v2',
			rewMode = 'orig',
			**kwargs
	):
		  
		self.quick_init(locals())
		self.hand_type = hand_type  
		SawyerPushEnv.__init__(
			self,
			tasks = tasks,
			hand_type = hand_type,
			**kwargs
		)
		#self.hand_init_pos = [-0.00434313 , 0.76608467 , 0.26081535]
		self.demo = False
		self.max_path_length = 120
		self.camera_name = 'angled_cam'
		self.info_logKeys = ['placingDist' , 'pickRew' , 'reachRew' , 'placeRew']
		self.rewMode = rewMode
	   
		self.action_space = Box(
			np.array([-1, -1, -1, -1]),
			np.array([1, 1, 1, 1]),
		)

	@property
	def model_name(self):
	   
		#return get_asset_full_path('sawyer_xyz/sawyer_pickPlace.xml')
		#self.reset_mocap_quat = zangle_to_quat(np.pi/2) #this is the reset_mocap_quat for wsg grippers
	
		#self.reset_mocap_quat = zangle_to_quat(-np.pi/2)

		init_quat = [1,0,0,1]
	   
		self.reset_mocap_quat = (Quaternion(axis= [1,0,0] , angle = -np.pi/2)*Quaternion(init_quat)).elements
		return get_asset_full_path('sawyer_xyz/sawyer_wsg_coffee.xml')

	def _reset_hand(self):
		for _ in range(10):
			self.data.set_mocap_pos('mocap', self.hand_init_pos)
			self.data.set_mocap_quat('mocap', self.reset_mocap_quat)
			self.do_simulation([-1,1], self.frame_skip)
	
	def step(self, action):
		
		#action = [0,0,0,1]
		if self.demo:
			if self.curr_path_length <=20:
				action = [0 , 1, -1, -1]

			elif self.curr_path_length <=40:
				action = [0,1,1,1]

			elif self.curr_path_length <=70:
				action = [0,0.9,-0.25,1]

			elif self.curr_path_length<=100:
				action = [0,-1 ,1 , -1]

			noise = 5*1e-1*np.random.uniform(-1,1 , size= 3)
			noise_4d = np.concatenate([noise , [0]])
			action = np.array(action) + noise_4d
			#object position after picking and placing coffe : [-0.00434313  0.76608467  0.26081535]

		self.set_xyz_action(action[:3])
		self.do_simulation([ action[-1], -action[-1]])
  
		self._set_goal_marker(self._state_goal)
		ob = self._get_obs()
		reward , reachReward , pickReward , placeReward , placingDist = self.compute_reward(action, ob)
		self.curr_path_length +=1
		if self.curr_path_length == self.max_path_length:
			done = True
		else:
			done = False
		return ob, reward, done,  OrderedDict({  'epRew' : reward , 'reachRew': reachReward , 'pickRew': pickReward , 'placeRew': placeReward , 'placingDist': placingDist})
	

	def change_task(self, task):

		task = {'goal': np.array([0, 1.0, 0.05]), 'height': 0.06, 'obj_init_pos':np.array([0, 0.6, 0.04])}
		self.grasp = False
		self.pickCompleted = False

		if len(task['goal']) == 3:
			self._state_goal = task['goal']
		else:
			self._state_goal = np.concatenate([task['goal'] , [0.02]])
		self._set_goal_marker(self._state_goal)

		if len(task['obj_init_pos']) == 3:
			self.obj_init_pos = task['obj_init_pos']
		else:
			self.obj_init_pos = np.concatenate([task['obj_init_pos'] , [0.02]])
		
		#self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget



	def render(self, mode = 'human'):

		if mode == 'human':
			im_size = 500 ; norm = 1.0
			self.set_goal_visibility(visible = True)
		elif mode == 'nn':
			im_size = self.image_dim ; norm = 255.0
		elif mode == 'vis_nn':
			im_size = self.image_dim ; norm = 1.0
		else:
			raise AssertionError('Mode must be human, nn , or vis_nn')
	   
		if self.camera_name == 'angled_cam':
		   
			image = self.get_image(width= im_size , height = im_size , camera_name = 'angled_cam').transpose()/norm
			image = image.reshape((3, im_size, im_size))
			image = np.rot90(image, axes = (-2,-1))
			final_image = np.transpose(image , [1,2,0])
			if 'nn' in mode:
				final_image = final_image[:48 ,10 : 74,:]
			# elif 'human' in mode:
			#     final_image = final_image[:285, 60: 440,:]

		if self.hide_goal:
		   self.set_goal_visibility(visible = False)
		return final_image

	def compute_reward(self, actions, obs):
					
		if isinstance(obs, dict):
		   
			obs = obs['state_observation']

		objPos = obs[3:6]
		rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
		fingerCOM  =  (rightFinger + leftFinger)/2       
	   
		placingGoal = self._state_goal
		graspDist = np.linalg.norm(objPos - fingerCOM)
		placingDist = np.linalg.norm(objPos - placingGoal)

		def reachReward():

			graspRew = -graspDist
			if np.linalg.norm(objPos[:2] - fingerCOM[:2]) < 0.02 and fingerCOM[2]<0.05:
			   
				self.grasp = True
			return graspRew 

		def pickRew():

			if self.pickCompleted:
				return 10
			elif self.grasp:
				if abs(0.07 - objPos[2])<0.1:
					self.pickCompleted = True

				return 1/(abs(0.07 - objPos[2])+1e-1)
			else:
				return 0

		def placeRew():

			if self.pickCompleted:
				return np.exp(-placingDist)
			else:
				return 0


		reachReward = reachReward()
		pickReward = pickRew()
		placeReward = placeRew()
		reward = reachReward + pickReward + placeReward
		return [reward , reachReward , pickReward , placeReward, placingDist]

	def log_diagnostics(self, paths = None, prefix = '', logger = None):

		from rllab.misc import logger
		#if type(paths[0]) == dict:
			
			# if isinstance(paths[0]['env_infos'][0] , OrderedDict):
			#     #For SAC
			#     for key in self.info_logKeys:
			#         nested_list = [[i[key] for i in paths[j]['env_infos']] for j in range(len(paths))]
			#         logger.record_tabular(prefix + 'max_'+key, np.mean([max(_list) for _list in nested_list]) )
			#         logger.record_tabular(prefix + 'last_'+key, np.mean([_list[-1] for _list in nested_list]) )



			
		#For TRPO
		for key in self.info_logKeys:
			#logger.record_tabular(prefix+ 'sum_'+key, np.mean([sum(path['env_infos'][key]) for path in paths]) )
			logger.record_tabular(prefix+'max_'+key, np.mean([max(path['env_infos'][key]) for path in paths]) )
			#logger.record_tabular(prefix+'min_'+key, np.mean([min(path['env_infos'][key]) for path in paths]) )
			logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths]) )
			logger.record_tabular(prefix + 'mean_'+key, np.mean([np.mean(path['env_infos'][key]) for path in paths]) )




	  