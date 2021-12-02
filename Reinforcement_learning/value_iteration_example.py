import numpy as np

init = [0, 0, 0, 0]
discount = 0.75

def reward(init):
	states = len(init)
	rewards = np.zeros((states, states))
	for i in range(states):
		if i == 0:
			states[i, i+1] = 1
		elif i == 3:
			states[i, i-1] = 10
		elif i ==1:
			states[i, i+1] = 1
			states[i, i-1] = 1
		else:
			states[i, i+1]=10
			states[i,i-1] =1
	return rewards

rewards = reward(init)
pritn(rewards)
