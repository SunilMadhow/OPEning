import numpy as np
from mdp import MDP
from tmis import TMIS2
from algs import *


# Build toy mdp
H = 5	
S = 2
A = 2

np.random.seed(0)

which_model = np.random.randint(0, 2, H)
r = np.random.choice([1/4,1/2,3/4,1], (H, S, A))
# print(r)

Pa = np.array([[[1, 0],[3/4, 1/4]], [[0, 1], [1/2, 1/2]]])
Pb = np.array([[[1, 0],[1/2, 1/2]], [[0, 1], [1/4, 3/4]]])

P = np.array([Pa if which_model[i] else Pb for i in range(0, H)])

pi = np.array([[0, 1] for i in range(0, H)])

M = MDP(H, S, A, P, r, [1/2, 1/2])

###########################

pi = np.array([[1, 1] for i in range(0, H)])

vpi = M.evaluate(pi)
vpi_mc = M.rollout(pi, 10000)[0]

print("vpi = ", vpi)
print("vpi_mc = ", vpi_mc)

Vstar = value_iteration(M)
print("vstar = ", np.average(Vstar[0][0]))
print("vstar = ", M.evaluate(Vstar[1]))

pibad = value_iteration(MDP(H, S, A, P, -1*M.r, [1/2, 1/2]))[1] 
print("vpibad = ", M.evaluate(pibad))

piucb = ucbvi(M, 50000, .01)
print("vpiucb = ", M.evaluate(piucb[1][-1]))