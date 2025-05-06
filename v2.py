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


s0 = 1
T = 6
K = s0
r = 0.01
sig = 0.3
N = 252*T  # nombre de pas pour T=6

# Génération de l'axe du temps
t = np.linspace(T, 0.01, 2*T, endpoint=True)  # j'évite t=0 car division par zéro sinon

# Calcul point par point
P0 = np.array([P0_TW(Ti, K, s0, r, sig) for Ti in t])
PD = np.array([PD_TW(Ti, K, s0, r, sig, max(1,int(N*Ti/T))) for Ti in t])  # max(1,...) pour éviter N=0
X = sig*np.sqrt(t)/2

# Tracé
plt.plot(t, P0, label='Continu')
plt.plot(t, PD, label='Discret')
plt.plot(t, X, label='Estimation rapide')
plt.xlabel('Temps restant T')
plt.ylabel('Prix')
plt.legend()
plt.title('Prix de l\'option asiatique selon Turnbull & Wakeman')
plt.grid()
plt.show()


###########################################
# Q5 : Simulation d'une trajectoire brownienne et estimation du prix par une méthode Monte Carlo
###########################################


