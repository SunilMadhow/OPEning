import math

from mdp import MDP
from tmis import *
from copy import deepcopy
import numpy as np

def value_iteration(M : MDP):

	# P = M.P
	# r = M.r
	H = M.H
	S = M.S
	A = M.A

	V = np.zeros((H+1)*S).reshape(H+1, S)
	pi = np.zeros(H*S).reshape(H,S)

	for t in reversed(range(0, H)):
		for s in range(0, S):
			Qsa = np.zeros(A)
			for a in range(0, A):
				Ev = np.dot(V[t+1],M.index_P(t, s, a))
				Qsa[a] = M.index_r(t, s, a) + Ev
			V[t, s] = np.amax(Qsa)
			pi[t, s] = np.argmax(Qsa)

	return (V, pi.astype(int))

def transition_statistics(D, H, S, A):
	k = len(D)
	nhsas_ = np.zeros(H*S*A*S).reshape((H, S, A, S))
	for h in range(0, H):
		for s in range(0, S):
			for a in range(0, A):
				for s_ in range(0, S):
					for i in range(0, k):
						nhsas_[h, s, a, s_] += (D[i, h][0] == s) and (D[i, h][1] == a and D[i, h][3] == s_)
	nhsa = np.sum(nhsas_, axis = 3)
	return (nhsa, nhsas_)

def ucbvi(M: MDP, k, δ): #want this to generate a dataset and a good policy. mostly just need dataset
	D = []
	pi = np.random.randint(0, 2, (M.H, M.S))
	nhsas_ = np.zeros(M.H*M.S*M.A*M.S).reshape((M.H, M.S ,M.A, M.S))
	M_ = deepcopy(M)

	T = k*M.H
	L = math.log(5*M.S*M.A*T/δ)
	C = 7*M.H*math.sqrt(L)

	for i in range(0, k):	
		Z = M.rollout(pi, 1)
		tau = Z[2][0]
		print(tau)
		for sars in tau:
			x = int(sars[0])
			y = int(sars[1])
			z = int(sars[3])
			nhsas_[x, z, z] = nhsas_[x, z, z] + 1
		nhsa = np.sum(nhsas_, axis = 3)
		bi = np.ones(M_.H*M_.S*M_.A)/nhsa

		M_.r = M_.r
		M_ = MDP()









