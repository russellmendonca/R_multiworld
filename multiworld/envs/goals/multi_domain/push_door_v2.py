#import joblib
import pickle
import numpy as np
tasks = [# Pushing 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([0, 0.81, 0.02]) , 'door_pos': np.array([0, 1.0,  0.22])} , 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([-0.10, 0.77 , 0.02])  ,  'door_pos': np.array([0, 1.0,  0.22])} , 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([0.10, 0.77 , 0.02]) , 'door_pos': np.array([0, 1.0,  0.22])} , 

		 #Door
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.22]) , 'padded_target_angle': np.array([0.52359 , 0, 0]) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) } ,  # 30 degrees
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.22]) , 'padded_target_angle': np.array([0.698132, 0 , 0] ) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) } ,# 40 degrees	
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.22]) , 'padded_target_angle': np.array([0.8726 , 0 ,0]) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) }     # 50 degrees
		]

fileName = 'multi_door_v2'
fobj = open(fileName +'.pkl' , 'wb')
pickle.dump(tasks, fobj)

