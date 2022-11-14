import numpy as np
from mdp import MDP
from tmis import TMIS
from algs import *
H = 5	
S = 2
A = 2

which_model = np.random.randint(0, 2, H)
r = np.random.choice([1/4,1/2,3/4,1], (H, S, A))
# print(r)

Pa = np.array([[[1, 0],[3/4, 1/4]], [[0, 1], [1/2, 1/2]]])
Pb = np.array([[[1, 0],[1/2, 1/2]], [[0, 1], [1/4, 3/4]]])

# Pa = np.array([[1, 0], [3/4, 1/4], [0, 1], [1/2, 1/2]])
# Pb = np.array([[1, 0], [1/2, 1/2], [0, 1], [1/4, 3/4]])

P = np.array([Pa if which_model[i] else Pb for i in range(0, H)])


print(r)
# print("P = ", P)

# def V_to_Q(Vpi, pi):
# 	Qpi = np.zeros(H*S*A).reshape((H, S, A))
# 	for h in range(0, H):
# 		for s in range(0, S):
# 			for a in range(0, A):
# 				Q[h, s, a] = 

pi = np.array([[1, 1] for i in range(0, H)])

M = MDP(H, S, A, P, r, [1/2, 1/2])

# print("rpi ", M.calc_rpi(pi))
# print("rpi[1] ", M.calc_rpi(pi)[1])

# pi = np.array([[1, 1] for i in range(0, H)])
# # print("pi = ", pi)
# out = M.rollout(pi, 20000)

# D = out[2]

# print("Monte-carlo estimated value of pi: ", out[0])

# print(M.calc_Ppi(pi))
# print("True value of pi: ", M.evaluate(pi))
# # Policy iteration

# est = TMIS(D, H, S, A, r)

# print(value_iteration(M))
# Z = value_iteration(M)
# pi1 = Z[1]
# print("pi1 = ", pi1)
# # # 
# print("Vpi1 = ", Z[0])
# print("Vpi1[0] average = ", (Z[0][0, 0] + Z[0][0, 1])/2)
# print("pi1 output value: ", M.evaluate(pi1))
P = ucbvi(M, 100000, .1)
piucb = P[1]
print("ucbvi output estimated function ", P[0])
print("ucbvi output true value ", M.evaluate(piucb))
print("solution ", M.evaluate(value_iteration(M)[1]))

# print("VI output Monte Carlo value estimate: ", M.rollout(pi1, 5000)[0])
# print("nhsa ", est.nhsa)
# Mc = MonteCarlo_Control(pi, M, 10)
# # D = Mc.go(5, 10)
# print("D = ", D)
# print("Montecarlo Qpi =", Mc.estimate(D))

# print("TMIS estimate ", est.estimate(pi))