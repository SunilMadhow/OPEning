import gym
import numpy as np
# from rlberry.agents import UCBVIAgent
from discrete_mc import *
from ucbviagent import *
rm = None #"human"
cenv = gym.make("MountainCar-v0", render_mode  = rm)
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


R = 0
# s = np.array(cenv2.reset())

s = env.reset()[0]
# print(env.)
print("s = ", s)
# for t in range(0, 20):
# 	ds = state_to_index(env, s)
# 	print("ds = ", ds)
# 	print("pi says ", pi[t, ds])
# 	for i in range(0, 10):
# 		s, r, done, info, _ = env.step(pi[t, ds])
# 		print()
# 		R = R + r
# 	if done: 
# 		print("finished")
# 		break
# print("R = ", r)
# print(continuous_to_discrete(env, (.54, .07)))

# print(discrete_to_continuous(env, (17, 14)))
# print("dd, ", discrete_dynamics(env, (7, 14), 2, 10))
# P = buildP(env)
# Z = value_iteration(env, P, env.r)

Z = ucbvi(env, 500, .01)
pi = Z[1]
print("rolling out last pol")
rollout(env, pi)
# print("Z = ", Z)
env.close()


