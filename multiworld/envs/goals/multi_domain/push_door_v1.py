#import joblib
import pickle
import numpy as np
tasks = [# Pushing 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([0, 0.81, 0.02]) ,  'door_pos': np.array([0, 1.0,  0.3])} , 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([-0.15, 0.77 , 0.02]) ,  'door_pos': np.array([0, 1.0,  0.3]) } , 
		 {'task': 'push' , 'obj_init_pos': np.array([0, 0.6 , 0.02]) , 'goal_pos': np.array([0.15, 0.77 , 0.02]) ,  'door_pos': np.array([0, 1.0,  0.3]) } , 

		 #Door
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.3]) , 'padded_target_angle': np.array([0.29 , 0, 0]) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) } , 
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.3]) , 'padded_target_angle': np.array([0.6, 0 , 0] ) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) } , 
		 {'task': 'door' , 'door_pos': np.array([0, 1.0,  0.3]) , 'padded_target_angle': np.array([0.87 , 0 ,0]) , 'obj_init_pos': np.array([0, 0.6 , 0.02]) } 
		]
#joblib.dump(tasks , 'push_door_v1.pkl')
fobj = open('push_door_v1.pkl' , 'wb')

pickle.dump(tasks, fobj)

