
import pickle
import numpy as np
from matplotlib import pyplot as plt

obj_init_pos = np.array([0.0, 0.6, 0.02])
door_pos 	 = np.array([0,   1.0, 0.3])

tasks = [] ; num_tasks = 3
#################  num_tasks goals of each ###################################################################################################

import pickle
old_tasks = pickle.load(open('multiDomain_pushDoorDrawer_10each.pkl' , 'rb'))
new_tasks = []

mapping = {0:7 , 1:8, 2:9, #pushing
			3:12,  4:15,  5:19 , #door
			6:22,  7:25, 8:29  # drawer
}
for i in range(9):
	new_tasks.append( old_tasks[mapping[i]])
#
# import pickle
# pickle.dump(new_tasks , open('multiDomain_pushDoorDrawer_3each.pkl' , 'wb'))



# import joblib
# tasks_pool = {
#             'tasks_pool': tasks,
#         }
# joblib.dump(tasks_pool , 'tasks_pool.pkl')