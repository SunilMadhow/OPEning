import numpy as np
import random
class MDP:
	def __init__(self, H, S, A, P, r, d_0):
		self.H = H
		self.S = S
		# self.d_0 = np.array(d_0)
		self.r = r
		self.A = A
		self.P = P

		
		self.d_0 = np.array(d_0)


	def calc_Ppi(self, pi):
		Ppi = np.zeros((self.H*self.S*self.S)).reshape((self.H, self.S, self.S))
		for h in range(0, self.H):
			for s in range(0, self.S):
				# print(np.array(self.index_P(h, s, pi[h, s])))
				r = self.index_P(h, s, pi[h, s])
				Ppi[h, s, :] = r.copy()
		return Ppi

	def calc_rpi(self, pi):
		rpi = np.zeros((self.H*self.S)).reshape((self.H, self.S))
		for h in range(0, self.H):
			for s in range(0, self.S):
				rpi[h, s] = self.index_r(h, s, pi[h, s])
		return rpi

	def evaluate(self, pi):
		D = np.zeros(self.H*self.S).reshape((self.H, self.S))
		Ppi = self.calc_Ppi(pi)
		d = self.d_0
		D[0] = d
		# print("d = ", d)
		for h in range(1, self.H):
			D[h] = d@Ppi[h - 1]
			d = D[h]


		# print("Marginal distributions: ", D)
		v = 0
		rpi = self.calc_rpi(pi)

		for h in range(0, self.H):
			# print(np.dot(D[h], rpi[h]))
			v = v + np.dot(D[h], rpi[h])

		return v

	def index_P(self, h, s, a):
		return self.P[h, s, a]

	def index_r(self, h, s, a):
		return self.r[h, s, a]

	def stoch_rew(self, h, s, a):
		return np.random.uniform(self.index_r(h, s, a) - .5, self.index_r(h, s, a) + .5)
	
	def rollout(self, pi, k): 
		D = []
		Re = []
		G = 0
		for j in range(k):
			s = np.random.choice(range(self.S), p = self.d_0)
			R = 0
			tau = []
			for h in range(0, self.H):
				a = pi[h,s]
				s_ = np.random.choice(range(self.S), p = self.index_P(h, s, a))
				# r = self.index_r(h, s, a)
				r = self.stoch_rew(h, s, a)
				tau.append((s, a, r, s_))
				s = s_
				
				R = R + r
			G += R
			Re.append(R)
			D.append(tau)
		return (G/k, np.array(Re), np.array(D, dtype="int,int, f, int").reshape((k, self.H)))

	def rollout_multipol(self, Pi, run_vec, readData = False, writeData = False, readFrom = "", saveTo = ""): #run_vec[i] is number of trajectories to collect according to Pi[i]
		if readData:
			D = np.load(readFrom + ".npy")
			return D

		R = int(np.sum(run_vec))
		D = []
		for i in range(1, len(Pi)):
			if i%1000 == 0:
				print("Gathering ", run_vec[0], " trajectories using policy ", i)
			F = self.rollout(Pi[i], int(run_vec[i]))
			D.append(F[2][0])
			# D = np.concatenate((D, F[2]), axis = 0)
		if writeData:
			np.save(saveTo, D)
		return np.array(D)

####### Gym Environment #########

import gym
from gym.envs.toy_text.utils import categorical_sample
from gym import spaces

class ToyEnv(gym.Env): # two-state, two-action nonstationary MDP
	S = 2
	A = 2
	Pa = np.array([[[1, 0],[3/4, 1/4]], [[0, 1], [1/2, 1/2]]])
	Pb = np.array([[[1, 0],[1/2, 1/2]], [[0, 1], [1/4, 3/4]]])

	def __init__(self, H, d_0, randomize = True):
		super(ToyEnv, self).__init__()
		self.H = H
		self.d_0 = d_0

		if randomize:
			self.r = np.random.choice([1/4,1/2,3/4,1], (self.H, self.S, self.A))

			which_model = np.random.randint(0, 2, H)
			self.P = np.array([self.Pa if which_model[i] else self.Pb for i in range(0, H)])

		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Discrete(2)
		self.reward_range = (1/4, 1)

		self.R = 0

		self.M = MDP(self.H, self.S, self.A, self.P, self.r, self.d_0)

		self.reset()

	def reset(self):
		self.s = categorical_sample(self.d_0, self.np_random)
		self.h = 0

		return self.s
		

	def step(self, a):
		pmf = self.P[self.h, self.s, a]
		self.s = categorical_sample(pmf, self.np_random)
		r = self.r[self.h, self.s, a]
		self.R = self.R + r
		self.h = self.h + 1
		done = False
		if self.h == self.H:
			done = True

		return self.s, r, done

	def rollout_multipol(self, Pi, epsilon = None): #epsilon is a hack because only det. policies are rep but need sarsa shadow
		D = []

		for i in range(len(Pi)):
			done = False
			pi = Pi[i]
			tau = []
			while not done:
				s = int(self.s)
				h = int(self.h)
				a = int(pi[h, s])
				if epsilon:
					if random.uniform(0, 1) > epsilon:
						a = random.choice([0, 1])
				s_, r, done =  self.step(a)
				tau.append((s, a, r, s_))
			self.reset()
			D.append(tau)
		return np.array(D, dtype="int,int, f, int")

	def evaluate(self, pi):
		self.M.evaluate(pi)



def gym_rollout(pi, k, env):
	D = []
	Re = []
	G = 0
	for j in range(k):
		env.reset()

		s = env.s
		R = 0
		tau = []
		d = False
		while not d:
			a = pi[env.h, s]
			s_, r, d = env.step(a)

			tau.append((s, a, r, s_))
			s = s_
			R = R + r
		G += R
		Re.append(R)
		D.append(tau)
	return (G/k, np.array(Re), np.array(D, dtype="int,int, f, int").reshape((k, env.H)))




