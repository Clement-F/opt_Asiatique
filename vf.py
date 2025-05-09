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

# plt.plot(t, W)
# plt.xlabel("Temps")
# plt.ylabel("W(i Dt)")
# plt.title("Trajectoire d'un mouvement brownien")
# plt.grid(True)
# plt.show()



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
#print(f"Prix de l'option asiatique par Monte Carlo : {P_MC[0]:.4f}")

#print(f"Intervalle de confiance à 90% : [{P_MC[1]:.4f}, {P_MC[2]:.4f}]")


#####################################
# Q7 : Estimation Monte-Carlo de P^Δt par variable de contrôle
####################################



def prix_geometrique_asiatique(S0, K, T, r, sigma_E, r_E):
    """
    Calcule e^(-r*T) * E[max(G_bar(T) - K, 0)] (cf poly)

    """
    d1 = (np.log(S0 / K) + (r_E + 0.5 * sigma_E**2) * T) / (sigma_E * np.sqrt(T))
    d2 = d1 - sigma_E * np.sqrt(T)
    return np.exp(-r * T) * (S0 * np.exp(r_E * T) * norm.cdf(d1) - K * norm.cdf(d2))


def PDt_MC_control(T, K, S0, r, sigma, N, M):
    """
    Estimateur Monte Carlo de l'option asiatique avec variable de contrôle.
    M : nombre de trajectoires
    N : nombre de pas de temps

    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    
    # Paramètres de la variable de contrôle 
    sigma_E = sigma / N * np.sqrt((N + 1) * (2 * N + 1) / 6)
    r_E = (r - 0.5 * sigma**2) * (N + 1) / (2 * N) + 0.5 * sigma_E**2

    payoffs = []
    controls = []

    for j in range(M):
        W = Brown_traj(T, N, sigma)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        
        S_bar = np.mean(S[1:])
        G_bar = np.exp(np.mean(np.log(S[1:])))
        
        Y = np.exp(-r * T) * max(S_bar - K, 0)
        Z = np.exp(-r * T) * max(G_bar - K, 0)

        payoffs.append(Y)
        controls.append(Z)

    payoffs = np.array(payoffs)
    controls = np.array(controls)
    
    Z_mean = np.mean(controls)
    Z_exact = prix_geometrique_asiatique(S0, K, T, r, sigma_E, r_E)

    P_control = np.mean(payoffs) -  Z_mean + Z_exact

    # Calcul de l'erreur et de l'intervalle de confiance à 90%
    std = np.std(payoffs - controls)
    z = norm.ppf(1 - 0.10 / 2) 
    error = z * std / np.sqrt(M)

    CI_up = P_control + error
    CI_Down = P_control -error

    return P_control, CI_Down, CI_up, error


# Test d'application :
T = 0.5
K = 1
S0 = 1
r = 0.01
sigma = 0.3
dt = 1 / 252
N = int(T / dt)  
M = 10000 

P_VC = PDt_MC_control(T, K, S0, r, sigma, N, M)
#print(f"Prix estimé par Monte Carlo avec variable de contrôle : {P_VC[0]:.4f}")

#print(f"Intervalle de confiance à 90% : [{P_VC[1]:.4f}, {P_VC[2]:.4f}]")


#####################################
# Q8 : Tracés de P^Δt,MC et de P^Δt,MC,ctrl 
####################################

#comment tracer les estimateurs P^Δt,MC et P^Δt,MC,ctrl en fonction du nombre de trajectoires ainsi que leurs intervalles de confiance asymptotiques à 90%. Ajouter également l'approximation de P^Δt,MC,ctrl par la formule de Turnbull & Wakeman (cas discret)

m = np.unique(np.logspace(1, 4, 100, dtype=int))  # de 10 à 10_000, 100 points

P_MC, CI_Down_MC, CI_Up_MC = [], [], []
P_VC, CI_Down_VC, CI_Up_VC = [], [], []
P_TW = []

for M_i in m:
    p_mc, ci_down_mc, ci_up_mc, _ = PDt_MC(T, K, S0, r, sigma, N, int(M_i))
    p_vc, ci_down_vc, ci_up_vc, _ = PDt_MC_control(T, K, S0, r, sigma, N, int(M_i))
    p_tw = PD_TW(T, K, S0, r, sigma, N)

    P_MC.append(p_mc)
    CI_Down_MC.append(ci_down_mc)
    CI_Up_MC.append(ci_up_mc)

    P_VC.append(p_vc)
    CI_Down_VC.append(ci_down_vc)
    CI_Up_VC.append(ci_up_vc)

    P_TW.append(p_tw)

P_MC = np.array(P_MC)
P_VC = np.array(P_VC)
P_TW = np.array(P_TW)



# plt.plot(m, P_MC, label='P^Δt,MC')
# plt.fill_between(m, CI_Down_MC, CI_Up_MC, alpha=0.2, label='IC 90% P^Δt,MC')

# plt.plot(m, P_VC, label='P^Δt,MC,ctrl')
# plt.fill_between(m, CI_Down_VC, CI_Up_VC, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')

# plt.plot(m, P_TW, label='P^Δt,TW')

# plt.xlabel('Nombre de trajectoires M')
# plt.ylabel('Prix')
# plt.legend()
# plt.title('Comparaison des estimateurs Monte Carlo pour une option asiatique')
# plt.xscale('log')
# plt.grid()
# plt.show()


#####################################
# Q9 : Tracé de P^Δt,MC,ctrl en fonction de K 
####################################

K_vals = np.linspace(0.01, 2, 50)
Prices_MC_ctrl = []
CI_up_ctrl = []
CI_down_ctrl = []
Prices_TW = []

for K in K_vals:
    price_ctrl, ci_down_ctrl, ci_up_ctrl, _ = PDt_MC_control(T, K, S0, r, sigma, N, M)
    Prices_MC_ctrl.append(price_ctrl)
    CI_up_ctrl.append(ci_up_ctrl)
    CI_down_ctrl.append(ci_down_ctrl)
    Prices_TW.append(PD_TW(T, K, S0, r, sigma, N))

# plt.close()  
# plt.figure(figsize=(10, 4)) 
# plt.plot(K_vals, Prices_MC_ctrl, label='P^Δt,MC,ctrl')
# plt.fill_between(K_vals, CI_down_ctrl, CI_up_ctrl, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')
# plt.plot(K_vals, Prices_TW, label='P^Δt,TW')
# plt.xlabel('Prix d\'exercice K')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique en fonction de K')
# plt.legend()
# plt.grid()
# plt.show()

# Différence
Prices_MC_ctrl = np.array(Prices_MC_ctrl)
Prices_TW = np.array(Prices_TW)

# plt.close() 
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, Prices_MC_ctrl - Prices_TW, label="P^Δt,MC,ctrl - P^Δt,TW", color="blue")
# plt.axhline(0, linestyle='--', color='gray')
# plt.xlabel("Prix d'exercice K")
# plt.ylabel("Différence")
# plt.title("Différence entre les deux estimateurs")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#####################################
# Q10 : Tracé de P^Δt,MC,ctrl en fonction de sigma
####################################
K = 1
sigma_vals = np.linspace(0.01, 0.8, 50)
Prices_MC_ctrl = []
CI_up_ctrl = []
CI_down_ctrl = []
Prices_TW = []

for sigma in sigma_vals:
    price_ctrl, ci_down_ctrl, ci_up_ctrl, _ = PDt_MC_control(T, K, S0, r, sigma, N, M)
    Prices_MC_ctrl.append(price_ctrl)
    CI_up_ctrl.append(ci_up_ctrl)
    CI_down_ctrl.append(ci_down_ctrl)
    Prices_TW.append(PD_TW(T, K, S0, r, sigma, N))

# plt.close()  
# plt.figure(figsize=(10, 4))
# plt.plot(sigma_vals, Prices_MC_ctrl, label='P^Δt,MC,ctrl')
# plt.fill_between(sigma_vals, CI_down_ctrl, CI_up_ctrl, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')
# plt.plot(sigma_vals, Prices_TW, label='P^Δt,TW')
# plt.xlabel('Volatilité σ')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique en fonction de σ')
# plt.legend()
# plt.grid()
# plt.show()

# Différence
Prices_MC_ctrl = np.array(Prices_MC_ctrl)
Prices_TW = np.array(Prices_TW)

# plt.close()  
# plt.figure(figsize=(10, 4)) 
# plt.plot(sigma_vals, Prices_MC_ctrl - Prices_TW, label="P^Δt,MC,ctrl - P^Δt,TW", color="blue")
# plt.axhline(0, linestyle='--', color='gray')
# plt.xlabel("Volatilité σ")
# plt.ylabel("Différence")
# plt.title("Différence entre les deux estimateurs")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#####################################
# Q11 : Tracé de P^Δt,TW en fonction de dt 
####################################
sigma = 0.3

K_vals = np.linspace(0.01, 2, 50)
Prices_TW_0 = []
Prices_TW_1_252 = []
Prices_TW_1_52 = []
Prices_TW_1_12 = []

for K in K_vals:
    Prices_TW_0.append(PD_TW(1, K, S0, r, sigma, 1000000))  # dt = 0
    Prices_TW_1_252.append(PD_TW(1, K, S0, r, sigma, 252))  # dt = 1/252
    Prices_TW_1_52.append(PD_TW(1, K, S0, r, sigma, 52))  # dt = 1/52
    Prices_TW_1_12.append(PD_TW(1, K, S0, r, sigma, 12))  # dt = 1/12

# plt.close() 
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, Prices_TW_0, label='Dt = 0 (cas continu)')
# plt.plot(K_vals, Prices_TW_1_252, label='Dt = 1/252 (un jour)')
# plt.plot(K_vals, Prices_TW_1_52, label='Dt = 1/52 (une semaine)')
# plt.plot(K_vals, Prices_TW_1_12, label='Dt = 1/12 (un mois)')
# plt.xlabel('Prix d\'exercice K')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique selon Turnbull & Wakeman en fonction de Δt')
# plt.legend()
# plt.grid()

# plt.xlim(0.8, 1.2)
# plt.ylim(min(Prices_TW_0 + Prices_TW_1_252 + Prices_TW_1_52 + Prices_TW_1_12) * 0.95,
#          0.25)

# plt.tight_layout()
# plt.show()


#####################################
# Q12 : Tracé de la différence P^Δt,MC,ctrl - P^Δt,TW pour différents dt 
####################################

K_vals = np.linspace(0.01, 2, 50)
Prices_diff_1_252 = []
Prices_diff_1_52 = []
Prices_diff_1_12 = []

for K in K_vals:
    Prices_diff_1_252.append(PDt_MC_control(1, K, S0, r, sigma, 252, M) - PD_TW(1, K, S0, r, sigma, 252))  # dt = 1/252
    Prices_diff_1_52.append(PDt_MC_control(1, K, S0, r, sigma, 52, M) - PD_TW(1, K, S0, r, sigma, 52))  # dt = 1/52
    Prices_diff_1_12.append(PDt_MC_control(1, K, S0, r, sigma, 12, M) - PD_TW(1, K, S0, r, sigma, 12))  # dt = 1/12

plt.close() 
plt.figure(figsize=(10, 4))
plt.plot(K_vals, Prices_diff_1_252, label='Dt = 1/252 (un jour)')
plt.plot(K_vals, Prices_diff_1_52, label='Dt = 1/52 (une semaine)')
plt.plot(K_vals, Prices_diff_1_12, label='Dt = 1/12 (un mois)')
plt.xlabel('Prix d\'exercice K')
plt.ylabel('Différence')
plt.axhline(0, linestyle='--', color='gray')
plt.title('Différence entre l\' estimateur Monte Carlo et l\' approximation de Turnbull & Wakeman en fonction de Δt')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()