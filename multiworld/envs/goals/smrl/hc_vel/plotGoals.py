from matplotlib import pyplot as plt
import pickle
import numpy as np


def plot_hcGoals(_file , num_goals = 20):


	data = pickle.load(open(_file+ '.pkl', 'rb'))
	data = [i['velocity'] for i in data][:num_goals]

	ys = np.ones( num_goals)
	plt.scatter( data , ys)
	plt.savefig(_file+'.png')



plot_hcGoals('hc_vel_mean1_std1_v2')