from mdp import *
from algs import *
env = ToyEnv(5, np.array([1/2, 1/2]))
# R = 0

print(env.P)
print(env.r)

# pi = np.array([[1, 1] for i in range(0, env.H)])
# avg_reward, rewards, D =  gym_rollout(pi, 10, env)

# print("D = ", D)

D = sarsa(env, 10, .5, .32)
print(D)