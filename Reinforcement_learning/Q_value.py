"""
@version: Python 3.7
@author: Huihui Xu
@email: huihui.xu@pitt.edu
"""

import numpy as np
import pandas as pd

# generate transitional prob matrix when taking M
def prob_trans(states):
	"""
	Args: 
		states: state space

	Return:
		prob_matrix: probability transition matrix
	"""

	prob_matrixM = np.zeros((len(states), len(states)))
	prob_matrixC = np.zeros((len(states), len(states)))
	for state in states:
		if state == 1 or state == 2 or state == 3:
			prob_matrixM[state, state-1] =1
			prob_matrixC[state, state+2] = 0.7
			prob_matrixC[state, state] = 0.3
		elif state == 0:
			prob_matrixM[state, state]=1
			prob_matrixC[state, state]=1
		else:
			prob_matrixM[state, state-1]=1
			prob_matrixC[state, state]=1

	return prob_matrixM, prob_matrixC

# generate rewards matrix
def reward(states):
	reward_matrix = np.zeros((len(states), len(states)))
	for state_x in states:
		for state_y in states:
			if state_x != state_y:
				reward_matrix[state_x, state_y] = (np.abs(state_y - state_x))**(1/3)
			elif state_x != 0 and state_x == state_y:
				reward_matrix[state_x, state_y] = (state_x+4)**(-0.5)
			else:
				continue
	return reward_matrix

# generate Q table
def Q_value(states, actions, prob_matrixM, prob_matrixC, reward_matrix, discount, iterations):
	"""
	Args:
		states: different states
		actions: types of actions
		prob_matrixM: probability matrix when taking "M" action
		prob_matrixC: probability matrix when taking "C" action
		reward_matrix: reward matrix
		discount: discount factor
		iterations: number of iterations

	Return:
		q_table: stored Q table
	"""	

	# initialize the Q table
	q_table_history = []
	q_table_history.append(np.zeros((len(states), len(actions))))

	for iteration in range(iterations):
		# update the Q table
		q_table = np.zeros((len(states), len(actions)))
		for state in states:		
			for action in actions:
				updated_q=0
				if action == "M": # index: 0
					# find the transition probability matrix
					current_probs = prob_matrixM[state, :]
				else: 
					current_probs = prob_matrixC[state, :]

				current_rewards = reward_matrix[state, :]
				# find all the possible states according to transition probaility matrix
				next_states = np.where(current_probs!=0)[0]
				
				for next_state in next_states:
					next_qs = q_table_history[iteration+1-1][next_state, :]
					max_next_qs = np.max(next_qs)

					reward = reward_matrix[state, next_state]
					reward_prob = current_probs[next_state]
					updated_q += reward_prob*(reward+discount*max_next_qs)

				if action == "C":
					q_table[state, 0] = updated_q
				else:
					q_table[state, 1] = updated_q

		q_table_history.append(q_table)
	# convert it to a data frame
	newest_q = q_table_history[-1]
	q_table_df = pd.DataFrame(newest_q, columns= actions)

	return q_table_df


if __name__ == "__main__":
	states = [state for state in range(6)]
	actions = ['C', 'M']
	discount = 0.6
	iterations = 1
	prob_matrixM, prob_matrixC = prob_trans(states)
	# print("=========transitional probability matrices M and C")
	# print(prob_matrixM)
	# print(prob_matrixC)
	reward_matrix = reward(states)
	# print("========reward matrix=======")
	# print(reward_matrix)
	q_table = Q_value(states, actions, prob_matrixM, prob_matrixC, reward_matrix, discount, iterations)
	print(q_table)
