
"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""


import numpy as np

# if agent chooses to stay at the location, such an action is successful with 1/2
## if agent is at the leftmost or rightmost grid, it ends up at its neighboring location with 1/2
## if agent is at the inner grid locations, it has a probability 1/4 ends up at either of neighboring locations
# if agent chooses to move, such an action is successful with 1/3, fail with 2/3
## if agent chooses to move left when at the leftmost, staying at the location with 1/2
## if agent chooses to move right when at the rightmost, staying at the location with 1/2 

def value(vec, rewards, left,right,stay,discount,iterations):
	"""
	Args:
		vec: one episode results
		rewards: reward for each cell
		left: probability matrix if agent chooses to move left
		right: probability matrix if agent chooses to move right
		stay: probability matrix if agent chooses to stay
		discount: discount factor

	Return:
		history: results of iterations
	"""
	history= [vec]
	for iteration in range(1, iterations):
		stay_value = 0
		left_value = 0
		right_value = 0
		current = []
		# agent has three options: left, right, stay
		for i in range(len(vec)):
			# choose to stay
			stay_value = np.sum(stay_probs[i,:]*(rewards[i] + discount*history[iteration-1]), dtype = np.float64)
			
			# choose to left
			left_value = np.sum(left_probs[i,:]*(rewards[i] + discount*history[iteration-1]), dtype = np.float64)
			# choose to right
			right_value = np.sum(right_probs[i,:]*(rewards[i] + discount*history[iteration-1]), dtype = np.float64)

			max_value = max([stay_value, left_value, right_value])
			current.append(max_value)
		
		history.append(np.array(current))

	return history


def transation_probs(vec):
	states = len(vec)
	# agent chooses to stay: transition probabilities matrix
	stay_probs = np.zeros((states,states), dtype = np.float64)
	# agent chooses to move left: transition probabilities matrix
	left_probs = np.zeros((states, states), dtype = np.float64)
	# agent chooses to move left: transition probabilities matrix
	right_probs = np.zeros((states, states), dtype = np.float64)
	for i in range(states):
		for j in range(states):
			# if agent chooses to stay
			if i == j:
				# the agent stays at the leftmost position
				if i==0:
					stay_probs[i,j+1] = 1/2
				# the agent stays at the rightmost position
				elif i==4:
					stay_probs[i,j-1] = 1/2
				# the agent stays at the inner position
				else:
					stay_probs[i,j-1] = 1/4
					stay_probs[i,j+1] = 1/4
				stay_probs[i,j] = 1/2

	# if agent chooses to go left
	for i in range(states):
		# if the agent stays at the leftmost position
		if i == 0:
			left_probs[i,i] = 1/2
			left_probs[i,i+1] = 1/2
		else:
			left_probs[i,i] = 2/3
			left_probs[i,i-1] = 1/3

	# if agent choose to go right
	for i in range(states):
		# if the agent stays at the rightmost position
		if i == 4:
			right_probs[i,i] = 1/2
			right_probs[i,i-1] = 1/2
		else:
			right_probs[i,i] = 2/3
			right_probs[i,i+1] = 1/3

	return stay_probs, left_probs, right_probs






if __name__ == "__main__":
	# initial value functions of states
	v0 = np.array([0, 0, 0, 0, 0])
	# rewards
	rewards = np.array([0, 0, 0, 0, 1])
	# discount factor
	discount = 0.5
	iterations = 201
	stay_probs, left_probs, right_probs = transation_probs(v0)
	# print("=======if agent chooses to stay=======")
	# print(stay_probs)
	# print("=======if agent chooses to move left=======")
	# print(left_probs)
	# print("=======if agent chooses to move right=======")
	# print(right_probs)
	history = value(v0, rewards, left_probs,right_probs,stay_probs, discount,iterations)
	print(history[-1])

