from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv
from matplotlib import pyplot as plt
import numpy as np
env = SawyerPushEnv()
env.reset()
for i in range(int(1e4)):
    action = [0,0,0]
    env.step(action)
    env.render()