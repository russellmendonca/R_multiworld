

import pickle

import numpy as np

from matplotlib import pyplot as plt

save_dir = '/home/russell/multiworld/multiworld/envs/goals/'


def read_goals(fileName):

	fobj = open(save_dir+fileName+'.pkl', 'rb')
	goals = pickle.load(fobj)

	#import IPython
	#IPython.embed()

   
	return goals


def gen_doorGoals(fileName):

	tasks = []
	for count in range(20):

		task={}

		task['door_init_pos'] = np.random.uniform([-.3,0.8, 0.2] ,[.3,1, 0.4])
		task['goalAngle'] = np.random.uniform([0], [1.5708])

		tasks.append(task)

	fobj = open(save_dir+fileName+'.pkl', 'wb')
	pickle.dump(tasks, fobj)
	fobj.close()


def visualize_doorPos(fileName, xRange = [-.3, .3], yRange=[0.8, 1.0]):

	tasks = np.array(read_goals(fileName))
	from matplotlib import pyplot as plt

	for i in range(len(tasks)):
		task = tasks[i]

		color = np.random.uniform(0,1, size=3)
		plt.annotate(xy = task['door_init_pos'][:2], s= 'O'+str(i), color=color)
		#plt.annotate(xy = task['goal'], s='G'+str(i), color=color)


	plt.xlim(xRange[0], xRange[1])
	plt.ylim(yRange[0], yRange[1])

	plt.savefig(fileName+'.png')

#gen_doorGoals('doorOpening_60X20X20')
#visualize_doorPos('doorOpening_60X20X20')


def gen_pickPlaceGoals_simple(fileName):

	tasks = []
	for count in range(20):

		task={}

		xs = np.random.uniform(-.1, .1, size=2)
		obj_y = np.random.uniform(0.5, 0.6, size = 1)
		goal_y = np.random.uniform(0.7, 0.8, size = 1)


		task['obj_init_pos'] = np.array([xs[0], obj_y ,  0.02])
		task['goal'] = np.array([xs[1], goal_y, 0.02])
		#task['goal'] = np.array([xs[0], obj_y+0.1, 0.02])
		task['height'] = 0.06

		tasks.append(task)


	fobj = open(save_dir+fileName+'.pkl', 'wb')
	pickle.dump(tasks, fobj)
	fobj.close()




def gen_pickPlaceGoals(fileName):

	tasks = []
	for count in range(20):

		task={}

		xs = np.random.uniform(-.1, .1, size=2)
		ys = np.random.uniform(0.5, 0.8, size = 2)

		task['obj_init_pos'] = np.array([xs[0], ys[0], 0.02])
		task['goal'] = np.array([xs[1], ys[1], 0.02])
		task['height'] = 0.06

		tasks.append(task)


	fobj = open(save_dir+fileName+'.pkl', 'wb')
	pickle.dump(tasks, fobj)
	fobj.close()





def modify(oldName, newName):

	tasks = read_goals(oldName)


	new_tasks = []

	
	for old_task in tasks:

		new_task={}
	   
		new_task['obj_init_pos'] = np.concatenate([old_task['obj_init_pos'], [0.02]])
		new_task['goal'] = np.concatenate([old_task['goal'], [0.02]]) 
		new_task['height'] = 0.1

		new_tasks.append(new_task)

	

	fobj = open(save_dir+newName+'.pkl', 'wb')
	pickle.dump(new_tasks, fobj)
	fobj.close()


def plotOrigDistances(fileName):

   

	tasks = np.array(read_goals(fileName))

	OrigDistances = [np.linalg.norm(tasks[i]['obj_init_pos'][:2] - tasks[i]['goal'][:2]) for i in range(len(tasks))]

	for i in range(len(OrigDistances)):
		plt.plot(np.arange(5), OrigDistances[i]*np.ones(5), label = 'Task'+str(i))
	
	plt.legend(ncol = 3)
	plt.savefig('origDistances/'+fileName)


def two_obj_pushing(fileName, numgoals = 20):

	all_tasks = []
	num_configs = 5
	num_goals_per_config = int(numgoals/num_configs)

	reg1_obsPos = np.random.uniform((-.1 , 0.5 ) , (-0.025, 0.6 ) , size = (num_configs , 2))
	reg2_obsPos = np.random.uniform((0.025 , 0.5) , (.1 , 0.6) , size = (num_configs , 2))

	for pos1 , pos2 in zip(reg1_obsPos , reg2_obsPos):

		if np.random.random() < 0.5:
			obj1 = pos1 ; obj2 = pos2
		else:
			obj1 = pos2 ; obj2 = pos1

		goals = np.random.uniform((-.1 , .7 ) , (.1 , .8) , size = (num_goals_per_config , 2))
		for goal in goals:
			task = {}
			task['obj1_init_pos'] = obj1 ; task['obj2_init_pos'] = obj2
			task['goal'] = goal
			all_tasks.append(task)

	fobj  = open(fileName+'.pkl', 'wb')
	pickle.dump(all_tasks , fobj)

def visualize_two_obj(fileName, xRange = [-.15, .15], yRange=[0.5, 0.8]):

	tasks = np.array(read_goals(fileName))
	from matplotlib import pyplot as plt

	for i in range(len(tasks)):
		task = tasks[i]

		color = np.random.uniform(0,1, size=3)
		plt.annotate(xy = task['obj1_init_pos'][:2], s= 'O1'+str(i), color=color)
		plt.annotate(xy = task['obj2_init_pos'][:2], s= 'O2'+str(i), color=color)
		plt.annotate(xy = task['goal'][:2], s='G'+str(i), color=color)

	plt.xlim(xRange[0], xRange[1])
	plt.ylim(yRange[0], yRange[1])

	plt.savefig(fileName+'.png')


#two_obj_pushing('pickPlace_Obj2_v1')
#visualize_two_obj('pickPlace_Obj2_v1')
def fixedDoor_diffAngles(numgoals = 20 , fileName = 'door_60deg_val'):

	taskList = []
	targetAngles = np.random.uniform(0, 1.0472 , 20)
	for angle in targetAngles:
		taskList.append({'goalAngle' : angle})
	plt.scatter(np.arange(numgoals) , targetAngles)
	plt.savefig(fileName+'.png')

	import pickle
	
	fobj = open(fileName+'.pkl' , 'wb' )
	pickle.dump(taskList , fobj)
	fobj.close()

#fixedDoor_diffAngles()


def push_fixedObj(fileName, numgoals = 40):

	all_tasks = []
	goals = np.random.uniform((-.1 , .7 ) , (.1 , .8) , size = (numgoals , 2))
	for goal in goals:
		task = {}
		task['obj1_init_pos'] = np.array([0, 0.52])
		task['goal'] = goal
		all_tasks.append(task)

	fobj  = open(fileName+'.pkl', 'wb')
	pickle.dump(all_tasks , fobj)

def rand_angles(fileName , numgoals = 40):
	goals = np.random.uniform( 0 , 2*np.pi , size = (numgoals))
	plt.scatter(np.arange(numgoals) , goals)
	plt.savefig(fileName+'.png')

	fobj = open(fileName+'.pkl' , 'wb')
	pickle.dump(goals , fobj)

#rand_angles('claw_2pi')





def visualize_push(fileName, xRange = [-.15, .15], yRange=[0.5, 0.8]):

	tasks = np.array(read_goals(fileName))
	from matplotlib import pyplot as plt

	for i in range(len(tasks)):
		task = tasks[i]

		color = np.random.uniform(0,1, size=3)
		plt.annotate(xy = task['obj1_init_pos'][:2], s= 'O'+str(i), color=color)
		plt.annotate(xy = task['goal'][:2], s='G'+str(i), color=color)

	plt.xlim(xRange[0], xRange[1])
	plt.ylim(yRange[0], yRange[1])

	plt.savefig(fileName+'.png')


def ring_goals(numgoals =40):

	radius = 2.0
	#theta_list = np.random.uniform(0 , np.pi , size = numgoals)
	theta_list = np.random.uniform(np.pi/4 , (3/4)*np.pi , size = numgoals)

	goals = np.array([[radius*np.cos(theta) , radius*np.sin(theta)] for theta in theta_list ])
	pickle.dump(goals, open('rad2_quat.pkl', 'wb'))
	plt.scatter(goals[:,0] , goals[:,1])

	plt.xlim(-2.1,2.1)
	plt.ylim(0,2.1)
	plt.savefig('rad2_quat.png')

def vis_goals_ant():
	goals = pickle.load(open('goals_ant.pkl' , 'rb'))
	import ipdb
	ipdb.set_trace()
	plt.scatter(goals[:,0] , goals[:,1])

	plt.xlim(-2.1,2.1)
	plt.ylim(0,2.1)
	plt.savefig('goals_ant.png')

#vis_goals_ant()

