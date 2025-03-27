import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

#================= definition brownian ======================
def brownian(T,N):
    Xi = np.random.uniform(size = N)
    Ni = ss.norm(0,T/N).ppf(Xi)
    WT = sum(Ni)
    return WT

def Brown_traj(T,N):
    W=[]
    Xi = np.random.uniform(size =N)
    Ni = ss.norm(0,T/N).ppf(Xi)
    W.append(0)
    for k in range (0,N):
        W.append(W[k]+Ni[k])
    return W
T = 50
N= 1000*T
# W = Brown_traj(T,N)
# t = np.linspace(0,T,N+1)

# print(brownian(T,N))

# print(W)
# plt.plot(t,W)
# plt.show()
#================= definition fonctions =====================
T=1
r=1
s=1

M1 = (np.exp(r*T)-1)/(r*T)
M2 = (2*np.exp((2*r+s*s)*T))/((r+s*s)*(2*r+s*s)*T*T) + 2/(r*T*T) * (1/(2*r + s*s) - np.exp(r*T)/(r+s*s))
rA = np.log(M1)/T
sA = np.sqrt(np.log(M2)/T - 2*rA)

s0 = 1
eps = 1

def SA_t(t):
    s = s0 * np.exp((rA - sA*sA/2 )*t + sA * brownian(t,10*t) )
    return s

def P0MC(T,N,K):

    S = []
    for k in range(1,N):
        S.append(max(0, SA_t(T)-K))

    q1 = np.exp(-rA*T)
    q2 =  1/N * eps* np.sum(S)
    return q1*q2

# def P0TW(T,N,d,K):
#     m = np.log(s0) + (rA -)

T = 10
N = 5
K = 2

# for k in range(10):
#     print(P0TW(T,N,K))