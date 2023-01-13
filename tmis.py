import numpy as np
from mdp import MDP

class TMIS2:
	def __init__(self, D, H, S, A, r):
		self.D = D
		self.S = S
		self.A = A
		self.H = H
		self.r = r

		self.nhsas = np.empty((len(self.D) + 1)*self.H*self.S*self.A*self.S, dtype = int).reshape(((len(self.D) + 1), self.H, self.S, self.A, self.S)) #first row is 0 (index into i to get nhsas after i trajectories)
		# self.nhsa = np.empty((len(self.D) + 1, self.H, self.S, self.A), dtype = int)
		self.rhsa = np.empty(((len(self.D) + 1), self.H, self.S, self.A))
		self.num_computed = 0
		self.num_computed_r = 0

		self.calc_nhsas(0)
		self.calc_rtot(0)

	def calc_nhsas(self, n): # how many of the trajectories to compute on
		# print("call to calc_nhsas with n = ", n)
		if n == 0:
			self.nhsas[0] = np.zeros((self.H, self.S, self.A, self.S))
			# self.nhsa = np.sum(self.nhsas, axis = 4)
		if self.num_computed >= n:
			# print("didn't compute")
			return
		else:
			# print("computing")
			for i in range(self.num_computed + 1, n + 1):
				# print("incorporating trajectory ", i)
				tau = self.D[i - 1]
				self.nhsas[i] = self.nhsas[i - 1]
				for z in range(0, self.H):
					
					self.nhsas[i, z, tau[z][0], tau[z][1], tau[z][3]] = 1 + self.nhsas[i - 1, z, tau[z][0], tau[z][1], tau[z][3]]
				self.num_computed = self.num_computed + 1
				
		self.nhsa = np.sum(self.nhsas, axis = 4)
		# print("nhsa = ", self.nhsa)

	def d_0_est(self, n):
		self.calc_nhsas(n) #get updated nhsas values
		d_0 = np.sum(self.nhsa[:n + 1, 0], axis = 2)/np.sum((np.sum(self.nhsa[:n + 1, 0], axis =2)), axis = 1)[:, None]
		return d_0

	def calc_rtot(self, n): #returns **total** reward coming
		self.calc_nhsas(n)

		if n == 0:
			self.rhsa[0] = np.zeros((self.H, self.S, self.A))
		if self.num_computed_r >= n:
			return
		else:			
			for i in range(self.num_computed_r + 1, n + 1):
				# print("incorporating trajectory ", i)
				tau = self.D[i - 1]
				self.rhsa[i] = self.rhsa[i - 1]
				for z in range(0, self.H):
					self.rhsa[i, z, tau[z][0], tau[z][1]] = tau[z][2] + self.rhsa[i - 1, z, tau[z][0], tau[z][1]]
				self.num_computed_r = self.num_computed_r + 1

	def r_est(self, n):
		self.calc_rtot(n)
		return self.rhsa[n]/self.nhsa[n]


	def calc_Phat(self, n):
		self.calc_nhsas(n)
		# Phat = np.zeros((self.H*self.S*self.A*self.S)).reshape((self.H, self.S, self.A, self.S))
		# for h in range(0, self.H):
		# 	for s in range(0, self.S):
		# 		for a in range(0, self.A):
		# 			for s_ in range(0, self.S):
		# 				if (self.nhsa[n, h, s, a] == 0): 
		# 					Phat[h, s, a, s_] = 0
		# 				else:
		# 					Phat[h, s, a, s_] = self.nhsas[n, h, s, a, s_]/self.nhsa[n, h, s, a]
		Phat = (self.nhsas[n]/self.nhsa[n, :, :, :, None])
		# print("Phat shape = ", Phat)
		return Phat

	def evaluate(self, pi, n):
		d_0 = self.d_0_est(n)[n]
		rhat = self.r_est(n)
		# print("d_0_est = ", d_0)
		# print("rhat[0] = ", rhat[0])
		# print("d_0 = ", d_0)
		# print("M_ -> d_0 = ", d_0)
		Phat = self.calc_Phat(n)
		# print("Phat = ", Phat)
		# print("M_ -> P = ", Phat)
		return MDP(self.H, self.S, self.A, Phat, rhat, d_0).evaluate(pi)




# class TMIS:
# 	def __init__(self, D, H, S, A, r):
# 		self.D = D
# 		self.S = S
# 		self.A = A
# 		self.H = H
# 		self.r = r

# 		self.k = len(D)

# 		z = self.calc_nhsa()
# 		self.nhsa = z[0]
# 		self.nhsas_ = z[1]
# 		self.d_0 = self.d_0_est()

# 		self.Phat = self.calc_Phat()
		

# 		self.M = MDP(self.H, self.S, self.A, self.Phat, self.r, self.d_0)
# 		# print("Phat = ", self.Phat)


# 	def calc_nhsa(self): #extremely crudely calculated, but good enough for toy example
# 		nhsas_ = np.zeros(self.H*self.S*self.A*self.S).reshape((self.H, self.S, self.A, self.S))
# 		for h in range(0, self.H):
# 			for s in range(0, self.S):
# 				for a in range(0, self.A):
# 					for s_ in range(0, self.S):
# 						for i in range(0, self.k):
# 							nhsas_[h, s, a, s_] += (self.D[i, h][0] == s) and (self.D[i, h][1] == a and self.D[i, h][3] == s_)
# 		nhsa = np.sum(nhsas_, axis = 3)
# 		return (nhsa, nhsas_)

# 	def d_0_est(self):
# 		return np.sum(self.nhsa[0], 1)/np.sum((np.sum(self.nhsa[0], 1)))

# 	def calc_Phat(self):
# 		Phat = np.zeros((self.H*self.S*self.A*self.S)).reshape((self.H, self.S, self.A, self.S))
# 		for h in range(0, self.H):
# 			for s in range(0, self.S):
# 				for a in range(0, self.A):
# 					for s_ in range(0, self.S):
# 						if (self.nhsa[h, s, a] == 0): 
# 							Phat[h, s, a, s_] = 0
# 						else:
# 							Phat[h, s, a, s_] = self.nhsas_[h, s, a, s_]/self.nhsa[h, s, a]
# 		return Phat

# 	def estimate(self, pi):
# 		return self.M.evaluate(pi)