from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerDoorOpenEnv(SawyerXYZEnv):

    def __init__(
            self,
            doorGrasp_low=None,
            doorGrasp_high=None,

            tasks=[{'goalAngle': [1.0472], 'door_init_pos': [0, 1, 0.3]}],

            goal_low=np.array([0]),
            goal_high=np.array([1.58825]),

            #hand_init_pos=(0, 0.4, 0.05),
            hand_init_pos = (0, 0.5, 0.3) ,
            fixed_door_pos = (0 , 1, 0.3),
            image = False,
            doorHalfWidth=0.2,
            mpl = 100,

            **kwargs
    ):

        self.quick_init(locals())

        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if doorGrasp_low is None:
            doorGrasp_low = self.hand_low
        if doorGrasp_high is None:
            doorGrasp_high = self.hand_high

        self.max_path_length = mpl

        self.doorHalfWidth = doorHalfWidth
        self.fixed_door_pos = np.array([0, 1,0.3])
        self.hand_init_pos = np.array(hand_init_pos)
        self.info_logKeys = ['angleDelta']
      
        import pickle
        #tasks = np.array(pickle.load(open('/home/russell/multiworld/multiworld/envs/goals/Door_60X20X20.pkl', 'rb'))) 
        #tasks = np.array(pickle.load(open('/root/code/multiworld/multiworld/envs/goals/Door_60X20X20.pkl', 'rb')))   

        self.tasks = np.array(tasks)
        self.num_tasks = len(tasks)

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        self.hand_and_door_space = Box(
            np.hstack((self.hand_low, doorGrasp_low)),
            np.hstack((self.hand_high, doorGrasp_high)),
            dtype=np.float32
        )

        self.goal_space = Box(goal_low, goal_high)

        self.observation_space = Dict([
            ('state_observation', self.hand_and_door_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space)

        ])

        # self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_door_open.xml')


    def step(self, action):

        self.set_xyz_action(action[:3])

        self.do_simulation([action[-1], -action[-1]])

        self._set_goal_marker()
        # The marker seems to get reset every time you do a simulation

        ob = self._get_obs()

        reward, doorOpenRew, angleDelta = self.compute_reward(action, ob)
        self.curr_path_length += 1

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, {'doorOpenRew': doorOpenRew, 'epRew': reward, 'angleDelta': angleDelta}

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_site_pos('doorGraspPoint')
        flat_obs = np.concatenate((e, b))

        doorAngle = self.data.get_joint_qpos('doorjoint')

        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=doorAngle

        )

    def render(self, mode = 'human'):
        self.image_dim = 84
        if mode == 'human':
            im_size = 500 ; norm = 1.0
            #self.set_goal_visibility(visible = True)
        elif mode == 'nn':
            im_size = self.image_dim ; norm = 255.0
        elif mode == 'vis_nn':
            im_size = self.image_dim ; norm = 1.0
        else:
            raise AssertionError('Mode must be human, nn , or vis_nn')

        
        image = self.get_image(width= im_size , height = im_size , camera_name = 'door_diff').transpose()/norm
        image = image.reshape((3, im_size, im_size))
        image = np.rot90(image, axes = (-2,-1))
        final_image = np.transpose(image , [1,2,0])
        if 'nn' in mode:
            final_image = final_image[:48 ,10 : 74,:]

        return final_image
   

    def _set_door_xyz(self, doorPos):

        self.model.body_pos[-1] = doorPos

    def sample_tasks(self, num_tasks):

        # task_idx = np.random.randint(0, self.num_tasks, size = )

        indices = np.random.choice(np.arange(self.num_tasks), num_tasks)

        return self.tasks[indices]

    def _set_goal_marker(self):

        angle = self._state_goal

        door_pos = self.door_init_pos

        # import ipdb
        # ipdb.set_trace()
       
        goal_x = door_pos[0] + self.doorHalfWidth * (1 - np.cos(angle))

        goal_y = door_pos[1] - self.doorHalfWidth * np.sin(angle)

        goalSitePos = np.array([goal_x, goal_y, door_pos[2]])

        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goalSitePos
        )

    def change_task(self, task):

        self._state_goal = task['goalAngle']
        if 'door_init_pos' in task.keys():
            self.door_init_pos = task['door_init_pos']
        else:
            self.door_init_pos = self.fixed_door_pos

      
        self._set_goal_marker() 

    def reset_arm_and_object(self):

        self._reset_hand()

        self._set_door_xyz(self.door_init_pos)

        self.curr_path_length = 0

    def reset_model(self, reset_arg = None):


        if reset_arg == None:
            task = self.sample_tasks(1)[0]
        else:
            assert type(reset_arg) == int
            task = self.tasks[reset_arg]



        self.change_task(task)
        self.reset_arm_and_object()
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def get_site_pos(self, siteName):

        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self, actions, obs):

        if isinstance(obs, dict):
            obs = obs['state_observation']

        doorGraspPoint = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2

        doorAngleTarget = self._state_goal

        graspDist = np.linalg.norm(doorGraspPoint - fingerCOM)

        graspRew = -graspDist

        def doorOpenReward(doorAngle):

            # angleDiff = np.linalg.norm(doorAngle - doorAngleTarget)

            doorRew = 0
            if graspDist < 0.1:

                if doorAngle <= doorAngleTarget:

                    doorRew = max(10 * doorAngle, 0)

                elif doorAngle > doorAngleTarget:
                    doorRew = max(10 * (doorAngleTarget - (doorAngle - doorAngleTarget)), 0)

            return doorRew
            #norm = 10* doorAngleTarget
            #return 10*(doorRew / norm)

        doorAngle = self.data.get_joint_qpos('doorjoint')

        doorOpenRew = doorOpenReward(doorAngle)

        #reward = graspRew + doorOpenRew
        
        reward = doorOpenRew

        angleDelta = abs(doorAngleTarget - doorAngle)

        return [reward, doorOpenRew, angleDelta]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()

        return statistics

    def log_diagnostics(self, paths=None, prefix='', logger=None):
        from rllab.misc import logger
        if type(paths[0]) == dict:
            if type(paths[0]) == dict:
            #For SAC
                # for key in self.info_logKeys:
                #     nested_list = [[i[key] for i in paths[j]['env_infos']] for j in range(len(paths))]
                #     logger.record_tabular(prefix + 'last_'+key, np.mean([_list[-1] for _list in nested_list]) )

            #For TRPO
                for key in self.info_logKeys:
                    logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths]) )
    

        else:
            pass
            # for i in range(len(paths)):
            #     prefix=str(i)
            #     for key in self.info_logKeys:
            #         #logger.record_tabular(prefix+ 'sum_'+key, np.mean([sum(path['env_infos'][key]) for path in paths[i]]) )
            #         #logger.record_tabular(prefix+'max_'+key, np.mean([max(path['env_infos'][key]) for path in paths[i]]) )
            #         #logger.record_tabular(prefix+'min_'+key, np.mean([min(path['env_infos'][key]) for path in paths[i]]) )
            #         logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths[i]]) )
