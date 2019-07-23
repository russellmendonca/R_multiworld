from matplotlib import pyplot as plt
import pickle
import numpy as np


def plot_1d_pointGoals(_file , num_goals = 100):


	fobj = open(_file+ '.pkl', 'wb')
	
	goals = np.random.normal(0,1, size = (num_goals))
	import ipdb ; ipdb.set_trace()


	pickle.dump(goals , fobj)

	plt.scatter( np.arange(num_goals) , goals)
	plt.savefig(_file+'.png')

def plot_2d_pointGoals(_file , num_goals = 100):

	fobj = open(_file+ '.pkl', 'wb')
	goals = np.random.normal(0,1, size = (num_goals , 2))
	pickle.dump(goals , fobj)

	plt.scatter( goals[:,0] , goals[:,1])
	plt.savefig(_file+'.png')

plot_1d_pointGoals('1d_point_mean1_v1')
#plot_2d_pointGoals('2d_point_mean1_v2')