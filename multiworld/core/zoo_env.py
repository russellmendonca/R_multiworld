from gym.spaces import  Dict
from gym.spaces import Box 
import numpy as np
from multiworld.core.wrapper_env import ProxyEnv
from maml_zoo.logger import logger
#import rllab.misc.logger as rllab_logger


class ZooEnv(ProxyEnv):

   
    def __init__(self, wrapped_env):
        self.quick_init(locals())
        super(ZooEnv, self).__init__(wrapped_env)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """

        self.current_task = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.current_task


    def log_diagnostics(self, paths, prefix=''):
        
        for key in self.info_logKeys:
            logger.logkv(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths]) )



   
    

