

import pickle

import numpy as np

from matplotlib import pyplot as plt

save_dir = '/home/russell/multiworld/multiworld/envs/goals/benchmarks/'


def read_goals(fileName):

	fobj = open(save_dir+fileName+'.pkl', 'rb')
	goals = pickle.load(fobj)

	#import IPython
	#IPython.embed()

   
	return goals


def gen_cheetahGoals(fileName , num_tasks = 100):

	velocities = abs(np.random.normal(1.0, 1.0, size=(num_tasks,)))
	tasks = [{'velocity': velocity} for velocity in velocities]
		
	fobj = open(save_dir+fileName+'.pkl', 'wb')
	pickle.dump(tasks, fobj)
	fobj.close()

gen_cheetahGoals('hc_vel_mean1_std1_v1')