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

	Q = np.zeros(H*S*A).reshape((H, S, A))
	for t in reversed(range(0, H)):
		for s in range(0, S):
			Qsa = np.zeros(A)
			for a in range(0, A):
				Ev = np.dot(V[t+1],M.index_P(t, s, a))
				Qsa[a] = min(M.index_r(t, s, a) + Ev, H)
			Q[t, s] = Qsa
			V[t, s] = np.amax(Qsa)
			# pi[t, s] = np.argmax(Qsa)
			pi[t, s] = np.random.choice(np.flatnonzero(Qsa == Qsa.max()))
	# print("Q = ", Q)

	return (V, pi.astype(int))


def val_func(M : MDP, pi):
	H = M.H
	S = M.S
	A = M.A

	V = np.zeros((H+1)*S).reshape(H+1, S)
	for t in reversed(range(0, H)):
		for s in range(0, S):
			V[t, s] = np.dot(V[t + 1], M.index_P(t, s, pi[t, s])) + M.index_r(t, s, pi[t, s])
	return (V, np.average(V[0])) #should be d-weighted avg


def ucbvi(M: MDP, k, δ, readData = False, writeData = False, readFrom = None, saveTo = None): #want this to generate a dataset and a good policy. mostly just need dataset
	if readData:
		D = np.load(readFrom + "_D.npy")
		Pi = np.load(readFrom + "_Pi.npy")
		print("Warning: M_ not available due to readData flag being True")
		return (D, Pi, M)

	D = []
	
	pi = np.random.randint(0, 2, (M.H, M.S))
	Pi = [pi]

	nhsas_ = np.zeros(M.H*M.S*M.A*M.S, dtype = 'f').reshape((M.H, M.S ,M.A, M.S))
	M_ = deepcopy(M)

	H = M.H
	S = M.S
	A = M.A

	# nhsa = np.zeros(H*S*A).reshape((H, S,A))

	T = k*M.H
	L = math.log(5*M.S*M.A*T/δ)
	C = 7*M.H*math.sqrt(L)

	for i in range(0, k):	
		if i%1000 == 0:
			print("i: ", i)

		Z = M.rollout(pi, 1)
		tau = Z[2][0]
		D.append(tau)

		for t in range(len(tau)):
			sars = tau[t]
			x = sars[0]
			y = sars[1]
			z = sars[3]
			nhsas_[t, x, y, z] = nhsas_[t, x, y, z] + 1
		nhsa = np.sum(nhsas_, axis = 3)

		# print("nhsas_ = ", nhsas_)
		# print("nhsa = ")
		bi = 2*H*np.sqrt(L*np.ones(M_.H*M_.S*M_.A).reshape(M_.H, M_.S, M_.A)/nhsa)
		# print("b = ", bi)	
		M_.r = M.r + bi
		for h in range(H):
			for s in range(S):
				for a in range(A):
					for s_ in range(S):
						if nhsa[h, s, a] == 0:
							M_.P[h, s, a, s_] = 0
						else:
							M_.P[h, s, a, s_] = nhsas_[h, s, a, s_]/nhsa[h, s, a]
		# print("number of visited state-action-horizon pairs: ", np.count_nonzero(nhsa))
		R = value_iteration(M_)
		# print("Value at iteration ", i, " = ", R[0])
		pi = R[1]
		# print("pi = ", pi)
		Pi.append(pi)

	D = np.array(D)
	Pi = np.array(Pi)
	if writeData:
		np.save(saveTo + "_D", D)
		np.save(saveTo + "_Pi", Pi)
	# print("D = ", np.array(D))
	return (D, Pi, M_)

import random
def sarsa(env, k, ε, α):
	H = env.H
	S = env.S
	A = env.A


	Q = np.zeros((H+1)*S*A).reshape((H+1, S, A))
	is_done = False

	D = []
	Pi = []

	for i in range(0, k):
		tau = []
		is_done = False

		s = env.s
		h = env.h
		if random.uniform(0, 1) > ε:
			a = np.argmax(Q[h, s])
		else:
			a = random.choice([0, 1])

		pi = np.zeros(H*S).reshape((H, S)) #defined by each iterated Q function
		pi[h] = np.argmax(Q[h], axis = 1)
		while not is_done:

			s_, r, is_done = env.step(a)

			if random.uniform(0, 1) > ε:
				a_ = np.argmax(Q[h + 1, s_])
			else:
				a_ = random.choice([0, 1])
			
			Q[h, s, a] = Q[h, s, a] + α*(r + Q[h + 1, s_, a_] - Q[h, s, a])

			tau.append((s, a, r, s_))
			
			if env.h < env.H - 1:
				pi[h + 1] = np.argmax(Q[h + 1], axis = 1)
			a = a_
			s = s_
			h = env.h
		print("pi = ", pi.astype(int))

		D.append(tau)
		Pi.append(pi.astype(int))
		env.reset()
	return (np.array(D, dtype = "int,int, f, int"), Pi)


# def gym_ucbvi(env, k, δ): #want this to generate a dataset and a good policy. mostly just need dataset
# 	D = []
	
# 	pi = np.random.randint(0, 2, (M.H, M.S))
# 	Pi = [pi]

# 	nhsas_ = np.zeros(M.H*M.S*M.A*M.S).reshape((M.H, M.S ,M.A, M.S))
# 	M_ = deepcopy(M)

# 	H = M.H
# 	S = M.S
# 	A = M.A

# 	T = k*M.H
# 	L = math.log(5*M.S*M.A*T/δ)
# 	C = 7*M.H*math.sqrt(L)

# 	for i in range(0, k):	
# 		Z = gym_rollout(env, pi, 1)
# 		tau = Z[2][0]
# 		D.append(tau)
# 		print(tau)

# 		for t in range(len(tau)):
# 			sars = tau[t]
# 			x = sars[0]
# 			y = sars[1]
# 			z = sars[3]
# 			nhsas_[t][x, z, z] = nhsas_[t][x, z, z] + 1
# 		nhsa = np.sum(nhsas_, axis = 3)

# 		print("nhsas_ = ", nhsas_)
# 		bi = 2*H*np.sqrt(L*np.ones(M_.H*M_.S*M_.A).reshape(M_.H, M_.S, M_.A)/nhsa)
# 		print("b = ", bi)
# 		M_.r = M.r + bi
# 		for h in range(H):
# 			for s in range(S):
# 				for a in range(A):
# 					for s_ in range(S):
# 						if nhsa[h, s, a] == 0:
# 							M_.P[h, s, a, s_] = 0
# 						else:
# 							M_.P[h, s, a, s_] = nhsas_[h, s, a, s_]/nhsa[h, s, a]
# 		# print("number of visited state-action-horizon pairs: ", np.count_nonzero(nhsa))
# 		R = value_iteration(M_)
# 		# print("Value at iteration ", i, " = ", R[0])
# 		pi = R[1]
# 		# print("pi = ", pi)
# 		Pi.append(pi)
# 	# print("D = ", np.array(D))
# 	return (np.array(D), pi)
# 		# M_.P[h, s, a, s_] = nhsas_/nhsa









