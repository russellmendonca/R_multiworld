
import pickle
import numpy as np
from matplotlib import pyplot as plt

train_angles = np.linspace ( 0, np.pi/4, 10)

val_angles = np.linspace  ( 0, np.pi/4 , 50)

all_tasks = []
for angle in val_angles:
	all_tasks.append({ 'padded_target_angle': np.array([angle, 0, 0 ]) , 'task_family_idx': 1,  'door_pos': np.array([0. , 1. , 0.3]), \
		'task': 'door', 'obj_init_pos': np.array([0.  , 0.6 , 0.02])})

pickle.dump( all_tasks,  open('door_50val.pkl' , 'wb'))
#plt.scatter(np.ones(10) , train_angles, label = 'train_angles' )
#plt.scatter(np.ones(10) , val_angles , label = 'val angles')
#plt.legend()
#plt.savefig('door_val.png')
