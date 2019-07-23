
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
# tasks = [# Pushing 
# 		 {'task': 'push_1' , 'obj1_init_pos': np.array([0.1, 0.65 , 0.02]) , 'goal_pos': np.array([0.1, 0.75, 0.02]) , 'obj2_init_pos': np.array([-0.1, 0.65 , 0.02])  } , 
# 		 {'task': 'push_2' , 'obj1_init_pos': np.array([0.1, 0.65 , 0.02]) , 'goal_pos': np.array([-0.1, 0.75, 0.02]) , 'obj2_init_pos': np.array([-0.1, 0.65 , 0.02])  },
# 		 {'task': 'push_1' , 'obj1_init_pos': np.array([0.1, 0.65 , 0.02]) , 'goal_pos': np.array([0.1, 0.75, 0.02]) , 'obj2_init_pos': np.array([-0.1, 0.65 , 0.02])  },
# 		 {'task': 'push_2' , 'obj1_init_pos': np.array([0.1, 0.65 , 0.02]) , 'goal_pos': np.array([-0.1, 0.75, 0.02]) , 'obj2_init_pos': np.array([-0.1, 0.65 , 0.02])  },
# 		]

obj_init_pos = np.array([0.0, 0.6, 0.02])
door_pos 	 = np.array([0,   1.0, 0.3])

tasks = []

num_tasks = 10

########################### Val tasks ############################################
xs = np.linspace(-0.1, 0.1, 10)
ys = np.random.uniform(0.7, 0.8, 10)
plt.scatter(xs, ys)
plt.savefig('multi_push_trainTasks.png')
for x,y in zip(xs, ys):
	tasks.append({'task': 'push', 'obj_init_pos': obj_init_pos, 'goal_pos': np.array([x,y, 0.02]),
				  'door_pos': door_pos, 'task_family_idx': 0})

for door_angle in np.linspace(0, np.pi/4, num_tasks):
	tasks.append({'task': 'door' , 'door_pos': door_pos , 'padded_target_angle': np.array([door_angle , 0, 0]) ,
				  'obj_init_pos': obj_init_pos  , 'task_family_idx' : 1})

for drawer_target in np.linspace(0, 0.15, num_tasks) :
	tasks.append({'task': 'drawer' , 'obj_init_pos': obj_init_pos , 'padded_target_pos': np.array([drawer_target, 0, 0]) ,
				  'door_pos': door_pos , 'task_family_idx' : 2})




#
fobj = open('multiDomain_pushDoorDrawer_'+str(num_tasks)+'each_train.pkl' , 'wb')
pickle.dump(tasks, fobj)
#
# import joblib
# tasks_pool = {
#             'tasks_pool': tasks,
#         }
# joblib.dump(tasks_pool , 'tasks_pool.pkl')