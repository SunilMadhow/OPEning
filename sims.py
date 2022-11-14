import numpy as np
from mdp import MDP
from tmis import TMIS
from algs import *
H = 5	
S = 2
A = 2

N = 200_000

old_data = True

which_model = np.random.randint(0, 2, H)
r = np.random.choice([1/4,1/2,3/4,1], (H, S, A))
# print(r)

Pa = np.array([[[1, 0],[3/4, 1/4]], [[0, 1], [1/2, 1/2]]])
Pb = np.array([[[1, 0],[1/2, 1/2]], [[0, 1], [1/4, 3/4]]])

P = np.array([Pa if which_model[i] else Pb for i in range(0, H)])

pi = np.array([[1, 1] for i in range(0, H)])

M = MDP(H, S, A, P, r, [1/2, 1/2])


# Gather UCB-adaptive data #

print("running ucbvi -----")
F_ucb = ucbvi(M, N, .01, readData = old_data, writeData = False)

D_ucb = F_ucb[0]
Pi_ucb = F_ucb[1]

###########################

# Gather shadow data #
print("gathering shadow data using ucbvi pols ----")
D_shadow = M.rollout_multipol(Pi_ucb, np.ones(len(Pi_ucb)), readData = old_data, writeData = False)
print(D_shadow)

###########################
import matplotlib.pyplot as plt

pi = Pi_ucb[-1]

vpi = M.evaluate(pi)

adaptive_estimates = []
shadow_estimates = []

y_ad = []
y_sh = []

print("forming estimates")

for n in range(100, N):
	if n % 100 == 0:
		print("n = ", n)

	D_n_ucb = D_ucb[0:n]
	D_n_shadow = D_shadow[0:n]

	tmis_ucb = TMIS(D_n_ucb, H, S, A, r)
	tmis_shadow = TMIS(D_n_shadow, H, S, A, r)

	vhat_ucb = tmis_ucb.estimate(pi)
	vhat_sh = tmis_shadow.estimate(pi)

	adaptive_estimates.append(vhat_ucb)
	shadow_estimates.append(vhat_sh)

	y_ad.append(n*(vpi - vhat_ucb))
	y_sh.append(n*(vpi - vhat_sh))

plt.plot(np.arange(100, N), y_ad)
plt.plot(np.arange(100, N), y_sh)
plt.show()








