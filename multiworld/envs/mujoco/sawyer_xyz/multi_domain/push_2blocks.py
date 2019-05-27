from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
	create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
import mujoco_py
from multiworld.envs.mujoco.cameras import *
from pyquaternion import Quaternion
#from mujoco_py.mjlib import mjlib

def zangle_to_quat(zangle):
	"""
	:param zangle in rad
	:return: quaternion
	"""
	return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, -1], angle= zangle)).elements

class Sawyer_MultiPushEnv( SawyerXYZEnv):
	def __init__(
			self,
			obj_low=None,
			obj_high=None,
			tasks = [{'task': 'push' , 'obj_pos': np.array([0, 0.52 , 0.02]) , 'goal_pos': np.array([0, 0.81]) }] , 
			#tasks = None,
			goal_low= np.array([-0.5 ,  0.4 ,  0.05 , None]),
			goal_high=np.array([0.5, 1. , 0.5 , None]),
			hand_init_pos = (0, 0.4, 0.05),
			doorHalfWidth=0.2,
			rewMode = 'posPlace',
			indicatorDist = 0.05,
			image = False,
			image_dim = 84,
			camera_name = 'angled_cam',
			mpl = 150,
			hide_goal = True,
			**kwargs
	):
		self.quick_init(locals())        
		SawyerXYZEnv.__init__(
			self,
			model_name=self.model_name,
			**kwargs
		)
		if obj_low is None:
			obj_low = self.hand_low
		if obj_high is None:
			obj_high = self.hand_high

		if goal_high is None:
			goal_high = self.hand_high
		if goal_low is None:
			goal_low = self.hand_low

		self.camera_name = camera_name
		self.objHeight = self.model.body_pos[-1][2]
		#assert self.objHeight != 0
		self.max_path_length = mpl
		self.image = image

		self.image_dim = image_dim
		self.tasks = np.array(tasks)

		self.num_tasks = len(tasks)
		self.rewMode = rewMode
		self.Ind = indicatorDist
		self.hand_init_pos = np.array(hand_init_pos)
		self.action_space = Box(
			np.array([-1, -1, -1]),
			np.array([1, 1, 1]),
		)
		self.hand_env_space = Box(
			np.hstack((self.hand_low, obj_low , obj_low)),
			np.hstack((self.hand_high, obj_high, obj_high)),
		)
		self.goal_space = Box(goal_low, goal_high)
		#self.initialize_camera(sawyer_pusher_cam)
		self.info_logKeys = ['placeDist', 'reachDist']
		#self.door_info_logKeys = ['angleDelta']

		self.hide_goal = hide_goal
		if self.image:
			self.set_image_obsSpace()

		else:
			self.set_state_obsSpace()

	def set_image_obsSpace(self):
		if self.camera_name == 'robotview_zoomed':
			self.observation_space = Dict([           
					('img_observation', Box(0, 1, (3*(48*64)+self.action_space.shape[0] , ), dtype=np.float32)),  #We append robot config to the image
					('state_observation', self.hand_env_space), 
				])
	def set_state_obsSpace(self):
		self.observation_space = Dict([           
				('state_observation', self.hand_env_space),
				('state_desired_goal', self.goal_space),
				('state_achieved_goal', self.goal_space)
			])

	def get_goal(self):
		return {            
			'state_desired_goal': self._state_goal,
	}
	  
	@property
	def model_name(self):
		#Remember to set the right limits in the base file (right line needs to be commented out)
		
		self.reset_mocap_quat = [1,0,1,0]
		return get_asset_full_path('sawyer_xyz/push_2blocks.xml')

		############################# WSG GRIPPER #############################
		#self.reset_mocap_quat = zangle_to_quat(np.pi/2) 
		#return get_asset_full_path('sawyer_xyz/sawyer_wsg_pickPlace_mug.xml')
		#return get_asset_full_path('sawyer_xyz/sawyer_wsg_pickPlace.xml')


	def step(self, action):

	
		
		self.set_xyz_action(action[:3])
		self.do_simulation([0,0])
		ob = self._get_obs()
	   
		reward , metrics  = self.compute_reward(action, ob)
		#import ipdb
		#ipdb.set_trace()
		#metrics = {}
		self.curr_path_length +=1
		if self.curr_path_length == self.max_path_length:
			done = True
		else:
			done = False

		
		return ob, reward, done, OrderedDict(metrics)

	def _get_obs(self):
		
		

		hand = self.get_endeff_pos()
		obj1Pos =  self.get_body_com("obj1")
		obj2Pos = self.get_body_com("obj2")
		
		flat_obs = np.concatenate((hand, obj1Pos , obj2Pos))

		if self.image:
			image = self.render(mode = 'nn')
			return dict(img_observation = np.concatenate([image.flatten() , hand]) , 
						state_observation = flat_obs)

		else:
			return dict(        
				state_observation=flat_obs,
				state_desired_goal=self._state_goal,        
				state_achieved_goal=self._state_goal,
			)

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

		#if self.camera_name == 'robotview_zoomed':
	   
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
   
	def _get_info(self):
		pass

	def _set_push_goal_marker(self):
		"""
		This should be use ONLY for visualization. Use self._state_goal for
		logging, learning, etc.
		"""
		goal = self._state_goal[:3]
		self.model.site_pos[self.model.site_name2id('goal')] = (
			goal[:3]
		)

	def set_goal_visibility(self , visible = False):

		# site_id = self.model.site_name2id('goal')
		# if visible:       
		#     self.model.site_rgba[site_id][-1] = 1
		# else:
		#     self.model.site_rgba[site_id][-1] = 0
		pass

	def sample_tasks(self, num_tasks):

		indices = np.random.choice(np.arange(self.num_tasks), num_tasks , replace = False)
		return self.tasks[indices]

	def reset_task(self, task):
		self.change_task(task)


	def _set_obj1_xyz(self, pos):
		qpos = self.data.qpos.flat.copy()
		qvel = self.data.qvel.flat.copy()
		qpos[9:12] = pos.copy()
		qvel[9:15] = 0
		self.set_state(qpos, qvel)

	def _set_obj2_xyz(self, pos):
		qpos = self.data.qpos.flat.copy()
		qvel = self.data.qvel.flat.copy()
		import ipdb
		ipdb.set_trace()

		qpos[9:12] = pos.copy()
		qvel[9:15] = 0
		self.set_state(qpos, qvel)

	
	def change_push_task(self, task):

		self._state_goal = task['goal_pos']
		
		if self.task_type == 'push_1':
			self.origPlacingDist = np.linalg.norm( self.obj1_init_pos[:2] - self._state_goal[:2])
		else:
			self.origPlacingDist = np.linalg.norm( self.obj2_init_pos[:2] - self._state_goal[:2])
		
		self.pickCompleted = False

	def change_task(self, task):

		self.task = task
		self.obj1_init_pos = np.array(task['obj1_init_pos'])
		self.obj2_init_pos = np.array(task['obj2_init_pos'])
		
		self._set_obj1_xyz(self.obj1_init_pos)
		#self._set_obj2_xyz(self.obj2_init_pos)
		
		self.task_type = task['task']
		self.change_push_task(task)

	
	def reset_agent_and_object(self):

		self._reset_hand()      
		self.curr_path_length = 0
		
	def reset_model(self, reset_arg= None):

		if reset_arg == None:
			task = self.sample_tasks(1)[0]
		else:
			assert type(reset_arg) == int
			task = self.tasks[reset_arg]

		self.current_task = task
		self.change_task(task)
		self.reset_agent_and_object()

		return self._get_obs()

	def _reset_hand(self):
		import time
		for _ in range(10):
			self.data.set_mocap_pos('mocap', self.hand_init_pos)
			self.data.set_mocap_quat('mocap', self.reset_mocap_quat)
			self.do_simulation(None, self.frame_skip)


	def get_site_pos(self, siteName):
		_id = self.model.site_names.index(siteName)
		return self.data.site_xpos[_id].copy()

	def compute_rewards(self, actions, obsBatch):
		#Required by HER-TD3
		assert isinstance(obsBatch, dict) == True
		obsList = obsBatch['state_observation']
		rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
		return np.array(rewards)

	def compute_reward(self, actions, obs):

		
		return self.compute_push_reward(actions, obs , self.task_type)

	
	def compute_push_reward(self, actions, obs , task_type):
		
		self._set_push_goal_marker()
		state_obs = obs['state_observation']

		endEffPos = state_obs[0:3] 

		if task_type == 'push_1':
			
			objPos = state_obs[3:6]
		else:
			objPos = state_obs[6:9]

		placingGoal = self._state_goal[:3]
		rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
		fingerCOM = (rightFinger + leftFinger)/2

		c1 = 1 ; c2 = 1
		reachDist = np.linalg.norm(objPos - fingerCOM)   
		placeDist = np.linalg.norm(objPos[:2] - placingGoal[:2])

		if self.rewMode == 'l2':
			reward = -reachDist - placeDist

		elif self.rewMode == 'l2Sparse':
			reward = - placeDist

		elif self.rewMode == 'l2SparseInd':
			if placeDist < self.Ind:
				reward = - placeDist
			else:
				reward = - self.origPlacingDist


		elif self.rewMode == 'posPlace':
			reward = -reachDist + 100* max(0, self.origPlacingDist - placeDist)

		
		metrics = {'reachDist': reachDist , 'placeDist': min(placeDist, self.origPlacingDist*1.5) , 'task': task_type}

		return [reward, metrics] 


	 
	def get_diagnostics(self, paths, prefix=''):
		statistics = OrderedDict()       
		return statistics

	def log_diagnostics(self, paths = None, prefix = '', logger = None):
		
		
		for key in self.info_logKeys:
			logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths]) )

	