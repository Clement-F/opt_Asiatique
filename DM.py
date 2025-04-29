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

def SA_t(t,r,sig,n):
    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig**2)*T))/((r+sig**2)*(2*r+sig**2)*T*T) + 2/(r*T*T) * (1/(2*r + sig**2) - np.exp(r*T)/(r+sig**2))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    s = s0 * np.exp((rA - sigA**2/2 )*t + sigA * brownian(t,n*t) )
    print('brown=',s)
    return s

def P0MC(T,N,K,r,sig,nMC):

    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig**2)*T))/((r+sig**2)*(2*r+sig**2)*T*T) + 2/(r*T*T) * (1/(2*r + sig**2) - np.exp(r*T)/(r+sig**2))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    S = []
    for k in range(1,nMC):
        S.append(np.maximum(0, SA_t(T,rA,sigA,N)-K))
    print(S)
    q1 = np.exp(-rA*T)
    q2 =  1/nMC * eps* np.sum(S)
    print(q1,q2)
    return q1*q2



def P0TW(T,K,s0,r,sig):
    M1 = (np.exp(r*T)-1)/(r*T)
    M2 = (2*np.exp((2*r+sig**2)*T))/((r+sig**2)*(2*r+sig**2)*T**2) + 2/(r*T**2) * (1/(2*r + sig**2) - np.exp(r*T)/(r+sig**2))
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)
    # print(M1,M2)
    # print(rA,sigA)
    # print(np.log(M2)/T - 2*rA)

    d = 1/(sigA*np.sqrt(T))*( np.log(s0/K) + rA*T)

    P = np.exp(-rA*T) * (s0*(ss.norm.cdf(d + (sigA*np.sqrt(T))*(sigA**2/2 *T))) - K *(ss.norm.cdf(d - (sigA*np.sqrt(T))*(sigA**2/2 *T) )))
    return P


def P0TWD(T,K,s0,r,sig,N):
    dt = T/N
    M1 = 1/N * (np.exp(r*dt)*(1-np.exp(r*N*dt))/(1-np.exp(r*dt)))
    M21 = 1/N**2 * (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*dt*N))/(1-np.exp((2*r+sig**2)*dt)))
    M22 = 1/N**2 * (  2*np.exp(r*dt)/(1-np.exp(r*dt))   *   (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*(N-1)*dt))/(1-np.exp((2*r+sig**2)*dt)) - (np.exp(((N+1)*r+sig**2)*dt) * (1-np.exp((r+sig**2)*(N-1)*dt))/(1-np.exp((r+sig**2)*dt))))  )
    M2 = M21 + M22
    rA = np.log(M1)/T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)
    # print(M1,M2)
    # print(rA,sigA)
    # print(np.log(M2)/T - 2*rA)
    
    d = 1/(sigA*np.sqrt(T))*( np.log(s0/K) + rA*T)

    P = np.exp(-r*T) * (s0* np.exp(-rA*T)*(ss.norm.cdf(d + (sigA*np.sqrt(T))*(sigA**2/2 *T))) - K *(ss.norm.cdf(d - (sigA*np.sqrt(T))*(sigA**2/2 *T) )))
    return P




s0=1
T = 6
K = s0
r=0.01
sig = 0.3
N=252*T
nm=100


M = P0MC(T,N,K,r,sig,nm)
print(P0TWD(T,K,s0,r,sig,N))
print(P0TW(T,K,s0,r,sig))
print(M)
# t = np.linspace(T,0,2*T,endpoint=False)
# P1 = P0TWD(t,K,s0,r,sig,N)
# P2 = P0TW(t,K,s0,r,sig)
# M = P0MC(T,N,K,r,sig)
# X = sig*np.sqrt(t)/2
# plt.plot(t,P1,label='price 2')
# plt.plot(t,P2,label='price')
# plt.plot(t,M,label="MC")
# plt.plot(t,X,label='esti')
# plt.legend()
# plt.show()