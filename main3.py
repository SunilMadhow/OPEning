from mdp import * 
from algs import *
from tmis import TMIS

env = ToyEnv(5, np.array([1/2, 1/2]))

## Collect UCB-VI dataset

F = gym_ucbvi(env, 1000, .01)
D = F[0]
Pi = F[1]

