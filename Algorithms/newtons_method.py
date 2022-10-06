# @author: Huihui Xu
# @version: Python 3.8

"""
Newton-Raphson method or the Newton-Fourier method is an interative process to solve
equation f(x) = 0. It begins with an initial guess and iteratively change the initial value
to solve f(x) = 0 when the x value converges. 

Key step: x_{n+1} = x_{n} - f(x_n)/f'(x_n)
Assumption: the inital value is close to the true solution; f(x) is differentiable at true solution
"""

import numpy as np
from sympy import * 


def newton(x0, f, symbol, offset=0.0001, steps=1000):
	"""
	x0: the initial value
	symbol: x
	f: unsolved equation
	"""
	step = 0
	result = 0
	x = Symbol(symbol)
	fprime = eval(f).diff(x)
	current_guess = x0
	def parser_f(x):
		function = eval(f)
		return function
	while step < steps:	
		upper = parser_f(x0)
		bottum = fprime.evalf(subs={x: current_guess})
		current_guess = x0 - upper/bottum
		if np.abs(x0 - current_guess) <= offset:
			return current_guess
		x0 = current_guess
		step +=1
	return current_guess


if __name__ == '__main__':
	f = "x**2 - 37"
	current_guess = newton(6, f, 'x')
	print(current_guess)

