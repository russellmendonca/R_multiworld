from matplotlib import pyplot as plt
import pickle
import numpy as np


def plot_1d_pointGoals(_file , num_goals = 100):


	fobj = open(_file+ '.pkl', 'wb')
	goals = np.random.normal(0,1, size = (num_goals, 1))
	goals = np.concatenate([ goals , np.zeros((100,1))], axis = 1)

	tasks = [{'goalPos': goal} for goal in goals]
	
	pickle.dump(tasks , fobj)

	plt.scatter( goals[:,0] , goals[:,1])
	plt.savefig(_file+'.png')

def plot_2d_pointGoals(_file , num_goals = 100):

	fobj = open(_file+ '.pkl', 'wb')
	goals = np.random.normal(0,1, size = (num_goals , 2))
	pickle.dump(goals , fobj)

	plt.scatter( goals[:,0] , goals[:,1])
	plt.savefig(_file+'.png')

def plot_pointGoals_circle(num_goals = 20, mean_rad = 0.5 , width = 0.1 , version = 100):

	_file = 'point_circle_mean'+str(int(mean_rad*100))+'_width'+str(int(width*100))+'_v'+str(version)

	rads = np.random.uniform(mean_rad - width , mean_rad + width , num_goals)
	thetas = np.random.uniform(0, np.pi , num_goals)
	
	goals = np.array([(rad*np.cos(theta) , rad*np.sin(theta)) for rad, theta in zip(rads, thetas)])
	tasks = [{'goalPos' : goal} for goal in goals ]

	fobj = open(_file+'.pkl' , 'wb')
	pickle.dump(tasks, fobj)

	plt.scatter( goals[:,0] , goals[:,1] )
	plt.savefig(_file+'.png')


def plot_goals(_file ):
	
	tasks = pickle.load(open(_file+'.pkl' , 'rb'))

	goals = [task['goalPos'] for task in tasks][:20]
	goals = np.array(goals)
	plt.scatter( goals[:,0] , goals[:,1] )
	plt.savefig(_file+'_custom.png')

def add_actionScale(_file):
	all_tasks = pickle.load(open(_file +'.pkl', 'rb'))
	for task in all_tasks:
		task['action_scale'] = np.random.uniform(1,2)

	pickle.dump(all_tasks , open(_file+'_aScale1_2.pkl' ,'wb'))

add_actionScale('point_circle_mean100_width10_v2')
#plot_goals('point_circle_mean100_width10_v1')


#plot_1d_pointGoals('1d_point_mean1_v1')
#plot_2d_pointGoals('2d_point_mean1_v2')

#plot_pointGoals_circle(num_goals = 100, mean_rad = 1 , width = 0 , version= 1)