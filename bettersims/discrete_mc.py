import gym
from gym.spaces import Discrete
import numpy as np
from copy import deepcopy

class DiscreteState(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array([10, 100])
		print(num_states)
		num_states = np.round(num_states, 0).astype(int) + 1
		print(num_states)
		self.num_states = num_states
		print("ns = ", num_states[0]*num_states[1])
		self.original = deepcopy(env.unwrapped)
		self.original.render_mode = None
		self.observation_space = Discrete(num_states[0]*num_states[1])

	def observation(self, obs):
		s = (obs - self.original.observation_space.low)*np.array([10, 100])
		s = np.round(s, 0).astype(int)
		return s

def continuous_to_discrete(env, obs):
	print(obs)
	return env.observation(obs)

def discrete_to_continuous(env, obs):
	s = env.original.observation_space.low + obs*np.array([1/10, 1/100])
	return s	

def discrete_dynamics(env, s, a, t = 10):
	s0 = discrete_to_continuous(env, s)
	env.original.state = s0
	# print("-> ", env.original.step(0))
	s_ = None
	for i in range(0, t):
		s_, r, done, truncated, _ = env.original.step(a)
		if done or truncated:
			break
	return env.observation(s_)
	# return s_

def state_to_index(env, s):
	return s[0]*env.num_states[1] + s[1]


# if __name__ == "__main__":
# 	pre_env = gym.make("MountainCar-v0")
# 	env = DiscreteState(pre_env)
# 	print(env.observation_space)


