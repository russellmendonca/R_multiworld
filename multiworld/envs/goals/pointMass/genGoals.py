import numpy as np
import pickle
from matplotlib import pyplot as plt


def gen_circle_goals(radMin  , radMax , numGoals):

	
	## Max possible radius distance that can be covered with mpl = 100 : 0.34
	fileName = str(numGoals)+'_goals_'+str(radMin)+'_'+str(radMax)+'_rad'
	rads = np.random.uniform(radMin , radMax , numGoals)
	thetas = np.random.uniform(0 , 2*np.pi , numGoals)

	tasks = [{'goalPos': (rad*np.cos(theta) , rad*np.sin(theta))} for rad , theta in zip(rads, thetas)]
	pickle.dump(tasks, open(fileName+'.pkl' , 'wb'))

	taskList = np.array([task['goalPos'] for task in tasks ])	
	plt.scatter(taskList[:,0] , taskList[:,1])
	plt.xlim(-0.33, 0.33)
	plt.ylim(-0.33, 0.33)
	plt.savefig(fileName+'.png')


radRanges = [(.2, .2) , (.15, .25) , (.10 , .30)]
goalDensities = [5, 10, 20, 40]

# for numGoals  in goalDensities:
# 	for radRange in radRanges:
# 		plt.clf()
# 		gen_circle_goals(radRange[0], radRange[1] , numGoals)