import numpy as np
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

		
		print("Marginal distributions: ", D)
		v = 0
		rpi = self.calc_rpi(pi)

		for h in range(0, self.H):
			# print(np.dot(D[h], rpi[h]))
			v = v + np.dot(D[h], rpi[h])

		return v

	def index_P(self, h, s, a):
		return self.P[h, s*self.A + a]

	def index_r(self, h, s, a):
		return self.r[h, s*self.A + a]
	
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
				r = self.index_r(h, s, a)
				tau.append((s, a, r, s_))
				s = s_
				
				R = R + r
			G += R
			Re.append(R)
			D.append(tau)
		return (G/k, np.array(Re), np.array(D, dtype="f,f, f, f").reshape((k, self.H)))