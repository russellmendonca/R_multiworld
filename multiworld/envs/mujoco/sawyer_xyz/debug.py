from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pickPlace import SawyerPickPlaceEnv
from matplotlib import pyplot as plt
import numpy as np
env = SawyerPickPlaceEnv()
env.reset()
for i in range(int(1e4)):
    action = np.random.uniform(-1,1 ,4)
    action[2] = -1
    env.step(action)
    env.render()