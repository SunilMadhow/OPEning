import math
import gym
from discrete_mc import *
from copy import deepcopy
import numpy as np

from itertools import product


## So gotta modify value_iteration so that it accepts a transition kernel instead of querying environment
## for dynamics. 
## Or do we? Since environment is determinisitic, I believe we can just directly do value_iteration on the
## true env with exploration-augmented rewards


def value_iteration(env, P, r):

    
    H = env.H #for mountaincar 
    S = env.observation_space.n
    print("S = ", S)
    A = env.action_space.n
    print("A = ", A)


    V = np.zeros((H+1)*S).reshape(H+1, S)

    # for j in range(0, env.num_states[1]):
    #     V[:, state_to_index(env, (env.num_states[0] - 1, j))] = 200
    pi = np.zeros(H*S).reshape(H,S)

    Q = np.zeros(H*S*A).reshape((H, S, A))
    for t in reversed(range(0, env.H)):
        print("t = ", t)
        for ds in product(range(0, env.num_states[0] - 1), range(0, env.num_states[1])): #first range encodes that V[top of the hill] = 0
            # print("s = ", state_to_index(env, ds)) 
            s = state_to_index(env, ds)
            Qsa = np.zeros(A)
            for a in range(0, A):
                Ev = np.dot(P[t, s, a], V[t + 1])
                Qsa[a] = -1 + Ev
                # if Ev > 100:
                #     print("sus ", (t, s, a))
            Q[t, s] = Qsa
            # print("Qsa ", Qsa)
            # print(np.amax(Qsa))
            V[t, s] = np.amax(Qsa)
            # pi[t, s] = np.argmax(Qsa)
            pi[t, s] = np.flatnonzero(Qsa == Qsa.max())[-1]

    for i in range(0, 20):
        print("V[", i, "] = ", V[i])
    return (V, pi.astype(int))

def buildP(env):
    P = np.zeros((env.H, env.S, env.A, env.S))
    for h in range(0, env.H):
        for ds in product(range(0, env.num_states[0]), range(0, env.num_states[1])):
            s = state_to_index(env, ds)
            for a in range(0, env.A):
                p = np.zeros(env.S)
                ds_ =  discrete_dynamics(env, ds, a)
                s_ = state_to_index(env, ds_)
                p[s_] = 1
                P[h, s, a] = p
    print("P = ", P[5, 250, 2])
    return P


def rollout(env, pi):
    s = env.reset()[0]
    # print(env.)
    tau = []

    for t in range(0, 20):
        ds = state_to_index(env, s)
        print("ds = ", ds)
        print("pi says ", pi[t, ds])
        for i in range(0, 10):
            s, r, done, info, _ = env.step(pi[t, ds])
        tau.append((ds, pi[t, ds], state_to_index(env, s)))
        if done: 
            print("finished")
            break 

# 
def ucbvi(env, k, delta):
    D = []

    pi = np.random.randint(0, 2, (env.H, env.observation_space.n))
    Pi = pi

    nhsas_ = np.zeros(env.H*env.S*env.A*env.S, dtype = 'f').reshape((env.H, env.S ,env.A, env.S))

    T = k*env.H
    L = math.log(5*env.S*env.A*T/δ)
    C = 7*env.H*math.sqrt(L)

    nhsa = np.zeros(H*S*A).reshape((H, S,A))

    for i in range(0, k):
        if i%1000 == 0:
            print("i = ", i)
        τ = rollout(env, pi)
        D.append(τ)
        for t in range(0, len(τ)):
            sars = τ[t]
            s = sars[0]
            a = sars[1]
            s_ = sars[2]
            nhsas_[t, x, y, z] = nhsas_[t, x, y, z] + 1
        nhsa = np.sum(nhsas_, axis = 3)
        bi = 2*H*np.sqrt(L*np.ones(env.H*env.S*env.A).reshape(env.H, env.S, env.A)/nhsa)

        pi = value_iteration(env, self.r + bi)[1]

        Pi.append(pi)

    return Pi
    # def ucbvi(M: MDP, k, δ, readData = False, writeData = False, readFrom = None, saveTo = None): #want this to generate a dataset and a good policy. mostly just need dataset
    #     if readData:
    #         D = np.load(readFrom + "_D.npy")
    #         Pi = np.load(readFrom + "_Pi.npy")
    #         print("Warning: M_ not available due to readData flag being True")
    #         return (D, Pi, M)

    #     D = []
        
    #     pi = np.random.randint(0, 2, (M.H, M.S))
    #     Pi = [pi]

    #     nhsas_ = np.zeros(M.H*M.S*M.A*M.S, dtype = 'f').reshape((M.H, M.S ,M.A, M.S))
    #     M_ = deepcopy(M)

    #     H = M.H
    #     S = M.S
    #     A = M.A

    #     # nhsa = np.zeros(H*S*A).reshape((H, S,A))

    #     T = k*M.H
    #     L = math.log(5*M.S*M.A*T/δ)
    #     C = 7*M.H*math.sqrt(L)

    #     for i in range(0, k):   
    #         if i%1000 == 0:
    #             print("i: ", i)

    #         Z = M.rollout(pi, 1)
    #         tau = Z[2][0]
    #         D.append(tau)

    #         for t in range(len(tau)):
    #             sars = tau[t]
    #             x = sars[0]
    #             y = sars[1]
    #             z = sars[3]
    #             nhsas_[t, x, y, z] = nhsas_[t, x, y, z] + 1
    #         nhsa = np.sum(nhsas_, axis = 3)

    #         # print("nhsas_ = ", nhsas_)
    #         # print("nhsa = ")
    #         bi = 2*H*np.sqrt(L*np.ones(M_.H*M_.S*M_.A).reshape(M_.H, M_.S, M_.A)/nhsa)
    #         # print("b = ", bi) 
    #         M_.r = M.r + bi
    #         for h in range(H):
    #             for s in range(S):
    #                 for a in range(A):
    #                     for s_ in range(S):
    #                         if nhsa[h, s, a] == 0:
    #                             M_.P[h, s, a, s_] = 0
    #                         else:
    #                             M_.P[h, s, a, s_] = nhsas_[h, s, a, s_]/nhsa[h, s, a]
    #         # print("number of visited state-action-horizon pairs: ", np.count_nonzero(nhsa))
    #         R = value_iteration(M_)
    #         # print("Value at iteration ", i, " = ", R[0])
    #         pi = R[1]
    #         # print("pi = ", pi)
    #         Pi.append(pi)

    #     D = np.array(D)
    #     Pi = np.array(Pi)
    #     if writeData:
    #         np.save(saveTo + "_D", D)
    #         np.save(saveTo + "_Pi", Pi)
    #     # print("D = ", np.array(D))
    #     return (D, Pi, M_)