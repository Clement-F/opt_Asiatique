import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.stats import norm


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

def Brown_traj(T, N, sigma=1):
    """
    Simule une trajectoire de mouvement brownien de variance sigma^2 sur [0,T] avec N pas.

    Paramètres :
    - T : temps final
    - N : nombre de pas (donc la trajectoire aura N+1 points)
    - sigma : écart-type du brownien (par défaut 1)

    """
    dt = T / N
    G = ss.norm.rvs(size=N)                      # N variables gaussiennes centrées réduites
    dW = sigma * np.sqrt(dt) * G                 # incréments du brownien
    W = np.concatenate(([0], np.cumsum(dW)))     # W_0 = 0, puis cumul des incréments
    return W


# Test d'application :
T = 0.5
sigma = 0.3
dt = 1 / 252
N = int(T / dt)

t = np.linspace(0,T,N+1)
W = Brown_traj(T, N)

plt.plot(t, W)
plt.xlabel("Temps")
plt.ylabel("W(i Dt)")
plt.title("Trajectoire d'un mouvement brownien")
plt.grid(True)
#plt.show()



def PDt_MC(T, K, S0, r, sigma, N, M):
    """
    Estime le prix d'une option asiatique P^Δt par méthode de Monte Carlo.

    Paramètres : 
    - T : temps d'échéance de l'option (en années)
    - K : prix d'exercice
    - S0 : prix initial de l'actif sous-jacent
    - r : taux d'intérêt sans risque
    - sigma : volatilité de l'actif sous-jacent
    - N : nombre de pas de temps
    - M : nombre de trajectoires Monte Carlo

    """
    dt = T / N
    payoffs = []

    for i in range(M):
        t = np.linspace(0, T, N+1)
        W = Brown_traj(T, N, sigma)  # Simulation d'une trajectoire brownienne
        
        # Calcul de la trajectoire de l'actif sous-jacent S(t)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W) 

        # Moyenne des prix sur la trajectoire 
        S_mean = np.mean(S[1:]) #on prend pas S0 car i commence à 1 

        # Payoff de l'option asiatique
        payoff = max(S_mean - K, 0)
        payoffs.append(payoff)
    
    # Actualisation de la moyenne des payoffs
    P = np.exp(-r * T) * np.mean(payoffs)

    # Calcul de l'erreur et de l'intervalle de confiance à 90%
    std = np.std(payoffs)
    z = norm.ppf(1 - 0.10 / 2) 
    error = z * std / np.sqrt(M)

    CI_up = P + error
    CI_Down = P -error

    return P, CI_Down, CI_up, error

# Test d'application :
T = 0.5
K = 1
S0 = 1
r = 0.01
sigma = 0.3
dt = 1 / 252
N = int(T / dt)  
M = 10000 

P_MC = PDt_MC(T, K, S0, r, sigma, N, M)
print(f"Prix de l'option asiatique par Monte Carlo : {P_MC[0]:.4f}")

print(f"Intervalle de confiance à 90% : [{P_MC[1]:.4f}, {P_MC[2]:.4f}]")


#####################################
# Q7 : Estimation Monte-Carlo de P^Δt par variable de contrôle
####################################


def prix_geometrique_asiatique(S0, K, T, r, sigma_E, r_E):
    """Prix exact de l'option asiatique géométrique (lognormale)"""
    d1 = (np.log(S0 / K) + (r_E + 0.5 * sigma_E**2) * T) / (sigma_E * np.sqrt(T))
    d2 = d1 - sigma_E * np.sqrt(T)
    return np.exp(-r * T) * (S0 * np.exp(r_E * T) * norm.cdf(d1) - K * norm.cdf(d2))

def PDt_MC_control(T, K, S0, r, sigma, N, M):
    """
    Estimateur Monte Carlo de l'option asiatique avec variable de contrôle.
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    
    # Paramètres de la variable de contrôle géométrique
    sigma_E = sigma * np.sqrt((2*N + 1) / (6 * (N + 1)))
    r_E = 0.5 * (r - 0.5 * sigma**2) + 0.5 * (r + 0.5 * sigma**2) * (N + 1) / (3 * N)

    payoffs = []
    controls = []

    for _ in range(M):
        W = Brown_traj(T, N, sigma)[1]
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        
        S_arith = np.mean(S[1:])
        S_geom = np.exp(np.mean(np.log(S[1:])))
        
        Y = np.exp(-r * T) * max(S_arith - K, 0)
        Z = np.exp(-r * T) * max(S_geom - K, 0)

        payoffs.append(Y)
        controls.append(Z)

    payoffs = np.array(payoffs)
    controls = np.array(controls)
    
    # Variable de contrôle
    Z_mean = np.mean(controls)
    Z_exact = prix_geometrique_asiatique(S0, K, T, r, sigma_E, r_E)

    # Estimateur corrigé
    b = 1  # ou covariance-based estimation possible
    P_control = np.mean(payoffs) - b * (Z_mean - Z_exact)

    return P_control