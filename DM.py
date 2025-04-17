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
s0 = 100
eps = 1

def SA_t(t,r,sig):
    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig*sig)*T))/((r+sig*sig)*(2*r+sig*sig)*T*T) + 2/(r*T*T) * (1/(2*r + sig*sig) - np.exp(r*T)/(r+sig*sig))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    s = s0 * np.exp((rA - sigA*sigA/2 )*t + sigA * brownian(t,10*t) )
    return s

def P0MC(T,N,K,r,sig):

    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig*sig)*T))/((r+sig*sig)*(2*r+sig*sig)*T*T) + 2/(r*T*T) * (1/(2*r + sig*sig) - np.exp(r*T)/(r+sig*sig))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    S = []
    for k in range(1,N):
        S.append(max(0, SA_t(T,r,sig)-K))

    q1 = np.exp(-rA*T)
    q2 =  1/N * eps* np.sum(S)
    return q1*q2



def P0TW(T,K,s0,r,sig):
    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig*sig)*T))/((r+sig*sig)*(2*r+sig*sig)*T*T) + 2/(r*T*T) * (1/(2*r + sig*sig) - np.exp(r*T)/(r+sig*sig))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    d = 1/(sigA*np.sqrt(T))*( np.log(s0/K) + rA*T)

    P = np.exp(-rA*T) * (s0*(ss.norm.cdf(d + (sigA*np.sqrt(T))*(sigA**2/2 *T))) - K *(ss.norm.cdf(d - (sigA*np.sqrt(T))*(sigA**2/2 *T) )))
    return P


def P0TWD(T,K,s0,r,sig,N):
    dt = T/N
    M1 = 1/N * (np.exp(r*dt)*(1-np.exp(r*N*dt)/(1-np.exp(r*dt))))
    M21 = 1/N**2 * (np.exp((2*r+sig*sig)*dt) * (1-np.exp((2*r+sig*sig)*dt*N))/(1-np.exp((2*r+sig*sig)*dt)))
    M22 = 1/N**2 * (  2*np.exp(r*dt)/(1-np.exp(r*dt))   *   (np.exp((2*r+sig*sig)*dt) * (1-np.exp((2*r+sig*sig)*(N-1)*dt))/(1-np.exp((2*r+sig*sig)*dt)) - (np.exp(((N+1)*r+sig*sig)*dt) * (1-np.exp((r+sig*sig)*(N-1)*dt))/(1-np.exp((r+sig*sig)*dt))))  )
    M2 = M21 + M22
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)
    print(M1,M2)
    print(rA,sigA)
    print(np.log(M2)/T - 2*rA)
    
    d = 1/(sigA*np.sqrt(T))*( np.log(s0/K) + rA*T)

    P = np.exp(-rA*T) * (s0*(ss.norm.cdf(d + (sigA*np.sqrt(T))*(sigA**2/2 *T))) - K *(ss.norm.cdf(d - (sigA*np.sqrt(T))*(sigA**2/2 *T) )))
    return P

s0=1
T = 6
K = s0
r=0.01
sig = 0.3
N=252*T

print(P0TWD(T,K,s0,r,sig,T))



# t = np.linspace(T,0,2*T,endpoint=False)
# P1 = P0TWD(t,K,s0,r,sig,N)
# # P2 = P0TW(t,K,s0,r,sig)
# X = sig*np.sqrt(t)/2
# print(P1)
# plt.plot(t,P1,label='price 2')
# # plt.plot(t,P2,label='price')
# plt.plot(t,X,label='esti')
# plt.legend()
# plt.show()