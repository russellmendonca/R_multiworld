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
            tasks = [{'goal': np.array([0, 0.9, 0.05]), 'height': 0.06, 'obj_init_pos':np.array([0, 0.6, 0.04])}] , 
            hand_type = 'weiss',
            liftThresh = 0.04,
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
        self.camera_name = 'angled_cam'
        self.info_logKeys = ['placingDist' , 'pickRew']
        self.rewMode = rewMode
        self.objHeight = 0.04
        self.heightTarget = self.objHeight + liftThresh
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
        
        self.set_xyz_action(action[:3])
        
        #For simple grasp
        grasp_control = np.random.uniform(-1,1)
        self.do_simulation([grasp_control , -grasp_control])
        # if self.get_endeff_pos()[-1] <= .06:
        #     print("should grasp")
        #     self.do_simulation([1,-1])
        # else:
        #     self.do_simulation([0, 0])
        
        # #For grasp and lift
        # if self.curr_path_length>=10:
        #     self.do_simulation([1,-1])
        # else:
        #     self.do_simulation([-1,1])


        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, ob)
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done,  OrderedDict({ 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist})
    

    def change_task(self, task):


        if len(task['goal']) == 3:
            self._state_goal = task['goal']
        else:
            self._state_goal = np.concatenate([task['goal'] , [0.02]])
        self._set_goal_marker(self._state_goal)

        if len(task['obj_init_pos']) == 3:
            self.obj_init_pos = task['obj_init_pos']
        else:
            self.obj_init_pos = np.concatenate([task['obj_init_pos'] , [0.02]])
        
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget



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
        heightTarget = self.heightTarget
        placingGoal = self._state_goal
        graspDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - placingGoal)

        def reachReward():

            graspRew = -graspDist
            #incentive to close fingers when graspDist is small
            if graspDist < 0.02:
                graspRew = -graspDist + max(actions[-1],0)/50
            return graspRew , graspDist

        def pickCompletionCriteria():

            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True



       

        def grasped():

            sensorData = self.data.sensordata
            return (sensorData[0]>0) and (sensorData[1]>0)

        def objDropped():

            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (graspDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits

        def orig_pickReward():
            
            hScale = 50
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
       
            elif (objPos[2]> (self.objHeight + 0.005)) and (graspDist < 0.1):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            
            hScale = 50

            if self.pickCompleted and grasped():
                return hScale*heightTarget
            elif (objPos[2]> (self.objHeight + 0.005)) and grasped() :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward(cond):
          
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))               
                placeRew = max(placeRew,0)
                return placeRew
            else:
                return 0
        #print(self.maxPlacingDist)
        reachRew, reachDist = reachReward()

        if self.rewMode == 'orig':
            pickRew = orig_pickReward()
            placeRew  = placeReward(cond = self.pickCompleted and (graspDist < 0.1) and not(objDropped()))        

        else:
            assert(self.rewMode == 'general')
            pickRew = general_pickReward()
            placeRew  = placeReward(cond = self.pickCompleted and grasped())
        

        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        #print(placingDist)
        return [reward, reachRew, reachDist, pickRew, placeRew, min(placingDist, self.maxPlacingDist)] 

    def log_diagnostics(self, paths = None, prefix = '', logger = None):

        from rllab.misc import logger
        if type(paths[0]) == dict:
            
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




        else:

            for i in range(len(paths)):
                prefix=str(i)
                for key in self.info_logKeys:
                    #logger.record_tabular(prefix+ 'sum_'+key, np.mean([sum(path['env_infos'][key]) for path in paths[i]]) )
                    logger.record_tabular(prefix+'max_'+key, np.mean([max(path['env_infos'][key]) for path in paths[i]]) )
                    #logger.record_tabular(prefix+'min_'+key, np.mean([min(path['env_infos'][key]) for path in paths[i]]) )
                    logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths[i]]) )
        

