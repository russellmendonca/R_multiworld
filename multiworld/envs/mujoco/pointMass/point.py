
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np

class PointEnv(MujocoEnv, Serializable):

    def __init__(self, tasks = [{'goalPos' : [0.2, 0] }] , init_pos = [0,0] , goal_pos = [0.2, 0], mpl = 200, *args, **kwargs):

        self.quick_init(locals())    
        
        model_name = get_asset_full_path('pointMass/point.xml')
        self.obj_init_pos = init_pos
       
        self.tasks = np.array(tasks)
        self.num_tasks = len(self.tasks)
        self.goalPos = self.tasks[0]['goalPos']
        self.curr_path_length = 0
        self.max_path_length = mpl

        MujocoEnv.__init__(self, model_name, frame_skip=1, automatically_set_spaces=True)
        #Serializable.__init__(self, *args, **kwargs)

        self.info_logKeys = ['targetDist']
        self.reset()

        # self.get_viewer()
        # self.viewer_setup()

    def _get_obs(self):
       
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def get_flat_obs(self):
       
        return self._get_obs()

    def viewer_setup(self):
        
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 10
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
    

    def get_site_pos(self, siteName):
       
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def step(self, action):
        
        action = np.clip(action , -1,1)
        self.do_simulation(action)
        
        ballPos = self.get_body_com("point")
        goalPos = self.goalPos
        obs = self._get_obs()
        #print(np.linalg.norm(ballPos - goalPos))

    
        reward = -np.linalg.norm(ballPos[:2] - goalPos[:2])
        self.curr_path_length +=1

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False

        return obs, reward, done, {'targetDist': -reward}

    def _set_obj(self, pos):
        
        
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0:2] = pos.copy()
        qvel[0:2] = 0
        self.set_state(qpos, qvel)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """       
        self.model.site_pos[self.model.site_name2id('goal')] = (
            goal[:3]
        )
    
    def sample_tasks(self, num_tasks):

        indices = np.random.choice(np.arange(self.num_tasks), num_tasks)
        return self.tasks[indices]

    def change_task(self, task):

        self.goalPos = task['goalPos']

    def reset_agent_and_object(self):
 
        self._set_obj(self.obj_init_pos)
        self.curr_path_length = 0

    def reset(self , reset_args = None):
        self.sim.reset()
        if reset_args == None:
            task = self.sample_tasks(1)[0]
        else:
            assert type(reset_args) == int
            task = self.tasks[reset_args]
       
        self.change_task(task)
        self.reset_agent_and_object()
        return self._get_obs()

    def log_diagnostics(self, paths = None, prefix = '', logger = None):

        for key in self.info_logKeys:
            logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths]) )
      