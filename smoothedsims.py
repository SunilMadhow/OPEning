import numpy as np
import math
from mdp import MDP
from tmis import TMIS2
from algs import *
H = 5	
S = 2
A = 2

N = 5000

# Build toy mdp

np.random.seed(0)

which_model = np.random.randint(0, 2, H)
r = np.random.choice([1/4,1/2,3/4,1], (H, S, A))
# print(r)

Pa = np.array([[[1, 0],[3/4, 1/4]], [[0, 1], [1/2, 1/2]]])
Pb = np.array([[[1, 0],[1/2, 1/2]], [[0, 1], [1/4, 3/4]]])

P = np.array([Pa if which_model[i] else Pb for i in range(0, H)])

M = MDP(H, S, A, P, r, [1/2, 1/2])

runs = 10
# Gather UCB-adaptive data #

print("running ucbvi -----")
D_ucb_list = []
Pi_ucb_list = []

for i in range(0, runs):

	F_ucb = ucbvi(M, N, .01, readData = False, writeData = False, readFrom = "ucbvi_10_000", saveTo = "ucbvi_10_000")

	D_ucb_list.append(F_ucb[0])
	Pi_ucb_list.append(F_ucb[1])
# print("D_ucb ", len(D_ucb))

D_shadow_list = []
print("gathering shadow data using ucbvi pols ----")
for i in range(0, runs):
	D_shadow = M.rollout_multipol(Pi_ucb_list[i], np.ones(len(Pi_ucb_list[i])), readData = False, writeData = False, readFrom = "ucbvi_sh_10_000", saveTo = "ucbvi_sh_10_000")
	D_shadow_list.append(D_shadow)
###########################
print("done_gathering shadow")
# Gather shadow data #

# print("D_sh ", len(D_shadow))

###########################
import matplotlib.pyplot as plt

pi = value_iteration(MDP(H, S, A, P, -1*M.r, [1/2, 1/2]))[1] 

vpi = M.evaluate(pi)

print("evaluating pi with true value ", vpi)

adaptive_estimates = []
shadow_estimates = []

Q = N

y_ad = np.zeros(Q)
y_sh = np.zeros(Q)
print("forming estimates")

for i in range(runs):
	tmis_ucb = TMIS2(D_ucb_list[i], H, S, A, r)
	tmis_shadow = TMIS2(D_shadow_list[i], H, S, A, r)
	print("run ", i)
	for n in range(1, Q):
		if n % 1000 == 0:
			print("n = ", n)

		vhat_ucb = tmis_ucb.evaluate(pi, n)
		vhat_sh = tmis_shadow.evaluate(pi, n)

		y_ad[n] = y_ad[n] + math.sqrt(n)*((vhat_ucb - vpi))
		y_sh[n] = y_sh[n] + math.sqrt(n)*((vhat_sh - vpi))

y_ad = y_ad/runs
y_sh = y_sh/runs

# print("min nhsa = ", np.amin(tmis_ucb.nhsa[n]))
print("Phat = ", tmis_ucb.calc_Phat(n))
# print(len(y_ad))
# np.save("adaptive", np.array(y_ad))
# np.save("shadow", np.array(y_sh))
# plt.rcParams["text.usetex"] = True
plot_step = 5
plt.plot(np.arange(1,Q)[::plot_step], np.zeros(len(y_ad))[::plot_step])
plt.plot(np.arange(1, Q)[::plot_step], y_ad[::plot_step], label = "adaptive")
plt.plot(np.arange(1, Q)[::plot_step], y_sh[::plot_step], label = "shadow dataset")
plt.title(str(runs) + "-Averaged Performance of TMIS estimator with Adaptivity Generated by UCB-VI")
plt.ylabel("$\sqrt{n}(\hat{v}^\pi - v^\pi)$")
plt.xlabel("n")
plt.legend()

plt.show()








