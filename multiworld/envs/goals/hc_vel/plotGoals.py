from matplotlib import pyplot as plt
import pickle
import numpy as np


def plot_hcGoals(_file , dist_id = 1 ,  num_goals = 100):


	goals = np.random.uniform(0 , dist_id , size = (num_goals,))
	tasks = [{'velocity': goal_vel} for goal_vel in goals]
	fobj = open(_file + '.pkl' , 'wb')
	pickle.dump(tasks, fobj)

	ys = np.ones( num_goals)
	plt.scatter( goals , ys)
	plt.savefig(_file+'.png')

def plot_spec_hcGoals(_file):


	goals = 0.5 * np.array(range(1,11))
	tasks = [{'velocity': goal_vel} for goal_vel in goals]
	fobj = open(_file + '.pkl' , 'wb')
	pickle.dump(tasks, fobj)

	#ys = np.ones( num_goals)
	#plt.scatter( goals , ys)
	#plt.savefig(_file+'.png')



plot_spec_hcGoals('hc_vel_inc0.5_max5')
