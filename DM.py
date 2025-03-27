import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

#================= definition brownian ======================
def brownian(T,N):
    Xi = np.random.uniform(N)
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
T = 5
N= 50
W = Brown_traj(T,N)
t = np.linspace(0,T,N+1)

print(W)
plt.plot(t,W)
plt.show()
#================= definition fonctions =====================
T=1
r=1
s=1

M1 = (np.exp(r*T)-1)/(r*T)
M2 = (2*np.exp((2*r+s*s)*T))/((r+s*s)*(2*r+s*s)*T*T) + 2/(r*T*T) * (1/(2*r + s*s) - np.exp(r*T)/(r+s*s))
rA = np.log(M1)/T
sA = np.sqrt(np.log(M2)/T - 2*rA)

s0 = 0
eps = 1

def SA_t(t,brownt):
    s = s0 * np.exp((rA - sA*sA/2 )*t + sA * brownt )
    return s

def P0TW(T):
    q1 = np.exp(-rA*T) 
    return q1