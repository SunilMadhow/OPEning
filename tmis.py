import numpy as np
from mdp import MDP




class TMIS:
	def __init__(self, D, H, S, A, r):
		self.D = D
		self.S = S
		self.A = A
		self.H = H
		self.r = r

		self.k = len(D)

		z = self.calc_nhsa()
		self.nhsa = z[0]
		self.nhsas_ = z[1]
		self.d_0 = self.d_0_est()

		self.Phat = self.calc_Phat()
		

		self.M = MDP(self.H, self.S, self.A, self.Phat, self.r, self.d_0)
		# print("Phat = ", self.Phat)


	def calc_nhsa(self): #extremely crudely calculated, but good enough for toy example
		nhsas_ = np.zeros(self.H*self.S*self.A*self.S).reshape((self.H, self.S, self.A, self.S))
		for h in range(0, self.H):
			for s in range(0, self.S):
				for a in range(0, self.A):
					for s_ in range(0, self.S):
						for i in range(0, self.k):
							nhsas_[h, s, a, s_] += (self.D[i, h][0] == s) and (self.D[i, h][1] == a and self.D[i, h][3] == s_)
		nhsa = np.sum(nhsas_, axis = 3)
		return (nhsa, nhsas_)

	def d_0_est(self):
		return np.sum(self.nhsa[0], 1)/np.sum((np.sum(self.nhsa[0], 1)))

	def calc_Phat(self):
		Phat = np.zeros((self.H*self.S*self.A*self.S)).reshape((self.H, self.S, self.A, self.S))
		for h in range(0, self.H):
			for s in range(0, self.S):
				for a in range(0, self.A):
					for s_ in range(0, self.S):
						if (self.nhsa[h, s, a] == 0): 
							Phat[h, s, a, s_] = 0
						else:
							Phat[h, s, a, s_] = self.nhsas_[h, s, a, s_]/self.nhsa[h, s, a]
		return Phat

	def estimate(self, pi):
		return self.M.evaluate(pi)