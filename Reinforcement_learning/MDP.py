
"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""


import numpy as np

# The reward matrix
def reward(init):
	states = len(init)
	rewards = np.zeros((states, states))
	for i in range(states):
		if i == 0:
			rewards[i, i+1] = 1
		elif i == 3:
			rewards[i, i-1] = 10
		elif i ==1:
			rewards[i, i+1] = 1
			rewards[i, i-1] = 1
		else:
			rewards[i, i+1]=10
			rewards[i,i-1] =1
	return rewards


def value(init, discount, actions, rewards, iterations):
	"""
	Args:
		init: initial vector of states
		discount: discount factor
		actions: types of actions 
		rewards: possible rewards when taking certain action
		iterations: number of iterations

	Return:
		history: store the history of value iterations
		optimal_actions: store all the optimal actions
	"""
	history = []
	history.append(init)
	optimal_actions = []
	action_counts = 0
	for iteration in range(1,iterations):
		# store optimal actions in each iteration
		iter_actions = []
		current_iter_values=[]
		for i in range(len(init)):
			
			if i >0 and i < 3: # current state and not ending points
				# next state after taking certain action
				states = [i + action for action in actions]
				
			elif i ==0: # if current state is at the initial, then only up is possible
				states = [i+1]
			else: # if current state is at the end, then only down is possible
				states = [i-1]

			current_rewards = [rewards[i, state] for state in states]

				# the value from the previous step
			pre_values = [discount*history[iteration-1][state] for state in states]
			current_values = np.array(current_rewards) + np.array(pre_values)

			max_current_value = np.max(current_values)	
			current_iter_values.append(max_current_value)
			optimal_action = np.argmax(current_values)
			iter_actions.append(optimal_action)

		history.append(current_iter_values)
		optimal_actions.append(iter_actions)
		if if_converge(history):
			print(f"converged after {len(history)-1} iterations ")
			return  history, optimal_actions
	return history, optimal_actions

def if_converge(history):
	if all(np.array(history[-1]) - np.array(history[-2])) <= 1e-5:
		return True
	return False

if __name__ == "__main__":
	init = [0, 0, 0, 0]
	discount = 0.75	
	rewards = reward(init)
	actions = [1, -1] # up and down
	iterations = 200
	history, optimal_actions = value(init, discount, actions, rewards, iterations)
	print(history[-1])
	print(optimal_actions[-1])
