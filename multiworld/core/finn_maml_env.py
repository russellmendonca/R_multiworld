from gym.spaces import  Dict
from gym.spaces import Box 
import numpy as np
from multiworld.core.wrapper_env import ProxyEnv
import rllab.misc.logger as rllab_logger


class FinnMamlEnv(ProxyEnv):

   
    def __init__(self, wrapped_env, reset_mode = 'index'):
        self.quick_init(locals())
        self._reset_args = None
        self.reset_mode = reset_mode
        super(FinnMamlEnv, self).__init__(wrapped_env)


    def sample_goals(self, num_goals):
      
        if self.reset_mode == 'index':
            return np.array(range(num_goals))
        elif self.reset_mode == 'task':
            return self.tasks
       
    #@overrides
    def reset(self, reset_args = None):

        self.sim.reset() 

        if reset_args is not None:
            self._reset_args = reset_args
     
        
        if self.reset_mode == 'index':
            if self._reset_args is None:
                 self._reset_args = 0
          
            reset_args = self.tasks[self._reset_args]

        elif self.reset_mode == 'task':
           
            if self._reset_args is None:
                 self._reset_args = self.tasks[0]
            reset_args = self._reset_args
        
        self.change_task(reset_args)
        #self.reset_arm_and_object()
        self.reset_agent_and_object()

        if self.viewer is not None:
            self.viewer_setup()

        return self.get_flat_obs()
   
  
    def log_diagnostics(self, paths, prefix='' , logger = None):

        if logger == None:
            logger = rllab_logger
        self.wrapped_env.log_diagnostics(paths = paths, prefix = prefix, logger = logger)
    
    #required by rllab parallel sampler
    def terminate(self):
        """
        Clean up operation,
        """
        pass
        
    

