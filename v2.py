import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


#####################################
# Q3 : Implémentation du prix d'une option asiatique selon Turnbull&Wakeman (cas continu)
####################################

def P0_TW(T, K, s0, r, sig):
    M1 = (np.exp(r*T) - 1) / (r*T)
    M2 = (2 * np.exp((2*r + sig**2)*T)) / ((r + sig**2)*(2*r + sig**2)*T**2) + 2 / (r*T**2) * (1/(2*r + sig**2) - np.exp(r*T)/(r + sig**2))

    rA = np.log(M1) / T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    d1 = (np.log(s0/K) + (rA + 0.5*sigA**2)*T) / (sigA * np.sqrt(T))
    d2 = d1 - sigA * np.sqrt(T)

    P = np.exp(-r*T) * (s0*np.exp(rA*T)*ss.norm.cdf(d1) - K*ss.norm.cdf(d2))
    return P



#####################################
# Q4 : Implémentation du prix d'une option asiatique selon Turnbull&Wakeman (cas discret)
####################################

def PD_TW(T, K, s0, r, sig, N):

    dt = T/N

    M1 = 1/N * np.exp(r*dt)*(1-np.exp(r*N*dt))/(1-np.exp(r*dt))

    M21 = 1/N**2 * (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*dt*N))/(1-np.exp((2*r+sig**2)*dt)))
    M22 = 1/N**2 * (  2*np.exp(r*dt)/(1-np.exp(r*dt))   *   (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*(N-1)*dt))/(1-np.exp((2*r+sig**2)*dt)) - (np.exp(((N+1)*r+sig**2)*dt) * (1-np.exp((r+sig**2)*(N-1)*dt))/(1-np.exp((r+sig**2)*dt))))  )
    M2 = M21 + M22


    rA = np.log(M1) / T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    d1 = (np.log(s0/K) + (rA + 0.5*sigA**2)*T) / (sigA * np.sqrt(T))
    d2 = d1 - sigA * np.sqrt(T)

    P = np.exp(-r*T) * (s0*np.exp(rA*T)*ss.norm.cdf(d1) - K*ss.norm.cdf(d2))
    return P


# s0 = 1
# T = 6
# K = s0
# r = 0.01
# sig = 0.3
# N = 252*T  # nombre de pas pour T=6

# # Génération de l'axe du temps
# t = np.linspace(T, 0.01, 2*T, endpoint=True)  # j'évite t=0 car division par zéro sinon

# # Calcul point par point
# P0 = np.array([P0_TW(Ti, K, s0, r, sig) for Ti in t])
# PD = np.array([PD_TW(Ti, K, s0, r, sig, max(1,int(N*Ti/T))) for Ti in t])  # max(1,...) pour éviter N=0
# X = sig*np.sqrt(t)/2

# Tracé
# plt.plot(t, P0, label='Continu')
# plt.plot(t, PD, label='Discret')
# plt.plot(t, X, label='Estimation rapide')
# plt.xlabel('Temps restant T')
# plt.ylabel('Prix')
# plt.legend()
# plt.title('Prix de l\'option asiatique selon Turnbull & Wakeman')
# plt.grid()
# plt.show()


#####################################
# Q5 : trajectoire brownienne et Monte-Carlo
####################################


def brownian(T,N):
    Xi = np.random.uniform(size = N)
    Ni = ss.norm(0,T/N).ppf(Xi)
    WT = sum(Ni)
    return WT

def Brown_traj(T,N,s=1):
    W=[]
    Ni = ss.norm(0,1).rvs(N)    #génère N variable gaussienne centrée réduite
    Ni *= s*np.sqrt(T/N)        #on transforme les N gaussiennes centrée pour qu'elles aient la variance souhaité
    W.append(0)                 
    W = np.cumsum(Ni)           #on crée le brownien de variance s en passant par la fonction cumsum
    return W

# T = 5           # temps de simulation
# detail =100     # détail de la trajectoire
# N= detail*T     # nombre de variable qui seront simulées

# W = Brown_traj(T,N)
# t = np.linspace(0,T,N+1)


# plt.plot(t,W)
# plt.show()

def Pdt_MC(r,S0,K,sig,N,T,n):
    
    LR = (r-sig**2/2)*T/n + sig*np.sqrt(T/n) * ss.norm(0,1).rvs((N,n))  #on simule les n morceaux des N trajectoire que prenne les logarithme des cours de l'action
    LRD = np.concatenate((np.ones((N,1))*np.log(S0),LR),axis=1)         #on ajoute le départ log(S0)
    
    Log_path = np.cumsum(LRD,axis=1)                                    #on colle les n morceaux pour obtenir des trajectoires
    Spath = np.exp(Log_path)                                            #on passe du logarithme du cours aux cours de l'option
    
    S_bar = np.mean(Spath[:,:-1],axis=1)                                #on prends la moyennes du cours
    payoff = np.exp(-r*T)*np.maximum(S_bar-K,0)                         #on calcule le gain de l'option asiatique
    
    price = np.mean(payoff)                                             #on prends la moyenne des N gains

    #calcul de l'erreur et de l'IC à 95%

    STD = np.std(payoff)
    error = 1.96 * STD/np.sqrt(N)
    CI_up = price + error
    CI_Down = price -error
    
    
    return price,STD,error,CI_up,CI_Down

# r=0.04
# K=so=100
# s=0.2
# T=1
# N=10**4
# n=10

# res = Pdt_MC(r, so, K, s, N, T,n)
# print("le prix estimé par MC est : ", res[0])
# print("la certitude est de 95%")
# print("l'intervalle de confiance à 95% est : [",res[4],",",res[3],"]")
# print("avec une erreur de ", res[2])




#####################################
# Q7 : Monte-Carlo par variable de controle
####################################


def Pdt_MC_ctrl(r,S0,K,sig,N,T,n):
    1+1
    return 0,0,0,0,0

#####################################
# Q8 : comparaison des estimations
####################################


# r=0.04
# K=so=100
# s=0.2
# T=1
# n_traj = 10

# n=10
# sample = [2**k for k in range(n)]

# P = np.zeros((2,n))
# err = np.zeros((2,n))
# up = np.zeros((2,n))
# down = np.zeros((2,n))

# lorem = 0

# for i in range(n) :
#     [P[0,i],lorem,err[0,i],up[0,i],down[0,i]] = Pdt_MC(r, so, K, s, sample[n], T,n_traj)
#     [P[1,i],lorem,err[1,i],up[1,i],down[1,i]] = Pdt_MC_ctrl(r, so, K, s, sample[n], T,n_traj)

# plt.plot(sample,P[0])
# plt.plot(sample,P[1])
# plt.show()

# plt.plot(sample,err[0])
# plt.plot(sample,err[1])
# plt.show()

# plt.plot(sample,up[0])
# plt.plot(sample,up[1])
# plt.plot(sample,down[0])
# plt.plot(sample,down[1])
# plt.show()

#####################################
# Q9 : 
####################################


#####################################
# Q11 : comparaison des deux estimations de TW pour différents strike à différentes échelles de temps
####################################

s0 = 1
T = 6
r = 0.01
sig = 0.3
n=10         #nombre d'échantillons du strike 

detail = [int(1e+6),252,52,12]
K = np.linspace(2,0,n,endpoint=False)
P = np.zeros((5,n))

for k in range(n):
    P[0,k] = ( P0_TW(T, K[k], s0, r, sig))
    print("pour dt = 1, le prix estimé par TW est :",P[0,k])

    for d in range(4) : 
        N = detail[d]*T
        P[d+1,k] = PD_TW(T, K[k], s0, r, sig, N)  
        t = np.linspace(T, 0, detail[d]*T, endpoint=False)
        print("pour dt = 1/",detail[d]," le prix estimé par TW est :",P[d+1,k])

plt.loglog(K,P[0]-P[1],label=0)
for k in range(2,5):
    plt.loglog(K,P[k]-P[1],label=k)

plt.legend()
plt.grid()
plt.show()

plt.plot(K,P[0])
plt.show()