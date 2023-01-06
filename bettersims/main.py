import gym
import numpy as np
# from rlberry.agents import UCBVIAgent
from discrete_mc import *
from ucbviagent import value_iteration

cenv = gym.make("MountainCar-v0", render_mode  = "human")
# cenv2 = gym.make("MountainCar-v0", render_mode = "human")
# cenv2 = cenv2.unwrapped
env = DiscreteState(cenv)
# env = env.unwrapped
s = env.reset()


# print("-> ", discrete_dynamics(env, (env.num_states - [2, 1]), 2))
# print("-> ", discrete_to_continuous(env, env.num_states - [1, 1]))
# cenv2.reset()
# cenv2.state = np.array(discrete_to_continuous(env, env.num_states - [2, 2]))
# # # cenv.state = cenv.high - [.]
# # print(cenv2.state)
# done = False
# for i in range(0, 50):
# 	s_, r, done, info, _ = cenv2.step(2)

print("done")
# s_ = env.step(1)[0]
# print("s = ", s)
# print("s_", s_)

Z = value_iteration(env, env.r)
pi = Z[1]
R = 0
# s = np.array(cenv2.reset())

s = env.reset()[0]
# print(env.)
print("s = ", s)
for t in range(0, 20):
	ds = state_to_index(env, s)
	print("ds = ", ds)
	print("pi says ", pi[t, ds])
	for i in range(0, 10):
		s, r, done, info, _ = env.step(pi[t, ds])
		print()
		R = R + r
	if done: 
		print("finished")
		break
print("R = ", r)

# print("Z = ", Z)
env.close()


