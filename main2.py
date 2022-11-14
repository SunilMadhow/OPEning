from mdp import *

env = ToyEnv(5, np.array([1/2, 1/2]))
R = 0
while True:
	s, r, d = env.step(env.action_space.sample())
	print("s = ", s)
	print("r = ", r)
	if d:
		env.reset()
	print()