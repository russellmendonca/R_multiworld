from matplotlib import pyplot as plt
import pickle
import numpy as np


def plot_recenteredGoals(taskFile , plotName ,  xRange = [-.15, .15], yRange=[0, 0.3]):

    taskList = pickle.load(open(taskFile , 'rb'))
    for i , task in enumerate(taskList):
        relPos = task['goal'][:2] - task['obj_init_pos'][:2]
        plt.annotate(xy = relPos, s='G'+str(i), color= np.random.uniform(0,1, size=3))

    plt.xlim(xRange[0], xRange[1])
    plt.ylim(yRange[0], yRange[1])

    plt.savefig(plotName+'.png')


plot_recenteredGoals('PickPlace_20X20_adj.pkl' , 'centered_pickPlace_20X20_adj.png')