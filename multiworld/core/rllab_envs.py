
from cached_property import cached_property
from multiworld.core.wrapper_env import ProxyEnv
from rllab.core.serializable import Serializable 


#from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box

from gym.spaces import Box as gymBox


def to_tf_space(space):
    if isinstance(space, gymBox):
        return Box(low=space.low, high=space.high)
    # elif isinstance(space, TheanoDiscrete):
    #     return Discrete(space.n)
    # elif isinstance(space, TheanoProduct):
    #     return Product(list(map(to_tf_space, space.components)))
    else:
        raise NotImplementedError

class EnvSpec(Serializable):

    def __init__(
            self,
            observation_space,
            action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        Serializable.quick_init(self, locals())
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

class TfEnv(ProxyEnv , Serializable):
    @cached_property
    def observation_space(self):
        return to_tf_space(self.wrapped_env.observation_space)
        #return self.wrapped_env.observation_space

    @cached_property
    def action_space(self):
        return to_tf_space(self.wrapped_env.action_space)
        #return self.wrapped_env.action_space

    @cached_property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
