

from multiworld.envs.mujoco.pointMass.point import PointEnv


env =PointEnv()

mpl = 200
for i in range(mpl):
    env.step(action = [-1,-1])
    print(env._get_obs()[:2])