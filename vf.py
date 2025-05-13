from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import time

###################################

def approx_norm_cdf(x):

    #déclaration des constantes
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    #calcul de l'approx
    t = 1/(1+b0*x)
    approx = 1/np.sqrt(2*np.pi) * np.exp(-1/2 * x**2)
    poly = b1*t + b2*t**2 + b3*t**3 + b4* t**4 + b5* t**5
    return 1-approx*poly



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

    P = np.exp(-r*T) * (s0*np.exp(rA*T)*approx_norm_cdf(d1) - K*approx_norm_cdf(d2))
    return P

# P0 = P0_TW(0.5, 1, 1, 0.01, 0.3)
# print(f"Approximation de Turnbull&Wakeman du prix de l'option asiatique (cas continu) : {P0:.4f}")


#####################################
# Q4 : Implémentation du prix d'une option asiatique selon Turnbull&Wakeman (cas discret)
####################################

def PD_TW(T, K, S0, r, sig, N):

    dt = T/N

    M1 = 1/N * np.exp(r*dt)*(1-np.exp(r*N*dt))/(1-np.exp(r*dt))

    M21 = 1/N**2 * (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*dt*N))/(1-np.exp((2*r+sig**2)*dt)))
    M22 = 1/N**2 * (  2*np.exp(r*dt)/(1-np.exp(r*dt))   *   (np.exp((2*r+sig**2)*dt) * (1-np.exp((2*r+sig**2)*(N-1)*dt))/(1-np.exp((2*r+sig**2)*dt)) - (np.exp(((N+1)*r+sig**2)*dt) * (1-np.exp((r+sig**2)*(N-1)*dt))/(1-np.exp((r+sig**2)*dt))))  )
    M2 = M21 + M22


    rA = np.log(M1) / T
    sigA = np.sqrt(np.log(M2)/T - 2*rA)

    d1 = (np.log(S0/K) + (rA + 0.5*sigA**2)*T) / (sigA * np.sqrt(T))
    d2 = d1 - sigA * np.sqrt(T)

    P = np.exp(-r*T) * (S0*np.exp(rA*T)*approx_norm_cdf(d1) - K*approx_norm_cdf(d2))
    return P

# Test d'application :
# T = 0.5
# sigma = 0.3
# dt = 1 / 252
# N = int(T / dt)
# P = PD_TW(T, 1, 1, 0.01, sigma, N)
# print(f"Approximation de Turnbull&Wakeman du prix de l'option asiatique (cas discret) : {P:.4f}")


#####################################
# Q5 : trajectoire brownienne et Monte-Carlo
####################################

def Brown_traj(T, N):
    """
    Simule une trajectoire de mouvement brownien de variance sigma^2 sur [0,T] avec N pas.

    Paramètres :
    - T : temps final
    - N : nombre de pas (donc la trajectoire aura N+1 points)
    - sigma : écart-type du brownien (par défaut 1)

    """
    dt = T / N
    G = ss.norm.rvs(size=N)                      # N variables gaussiennes centrées réduites
    dW = np.sqrt(dt) * G                         # incréments du brownien
    W = np.concatenate(([0], np.cumsum(dW)))     # W_0 = 0, puis cumul des incréments
    return W


# Test d'application :
T = 0.5
sigma = 0.3
dt = 1 / 252
N = int(T / dt)

# t = np.linspace(0,T,N+1)
# W = Brown_traj(T, N)

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
    payoffs = np.zeros(M)

    for i in range(M):
        t = np.linspace(0, T, N+1)
        W = Brown_traj(T, N)  # Simulation d'une trajectoire brownienne
        
        # Calcul de la trajectoire de l'actif sous-jacent S(t)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W) 

        # Moyenne des prix sur la trajectoire 
        S_mean = np.mean(S[1:]) #on prend pas S0 car i commence à 1 

        # Payoff de l'option asiatique
        payoffs[i] = max(S_mean - K, 0)
    
    # Actualisation de la moyenne des payoffs
    P = np.exp(-r * T) * np.mean(payoffs)

    # Calcul de l'erreur et de l'intervalle de confiance à 90%
    std = np.std(payoffs)
    error = 1.65 * std / np.sqrt(M)

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

# P_MC = PDt_MC(T, K, S0, r, sigma, N, M)
# print(f"Prix de l'option asiatique par Monte Carlo : {P_MC[0]:.4f}")

# print(f"Intervalle de confiance à 90% : [{P_MC[1]:.4f}, {P_MC[2]:.4f}]")


#####################################
# Q7 : Estimation Monte-Carlo de P^Δt par variable de contrôle
####################################


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

    payoffs = np.zeros(M)
    controls = np.zeros(M)

    for j in range(M):
        W = Brown_traj(T, N)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        
        S_bar = np.mean(S[1:])
        G_bar = np.exp(np.mean(np.log(S[1:])))
        
        Y = np.exp(-r * T) * max(S_bar - K, 0)
        Z = np.exp(-r * T) * max(G_bar - K, 0)

        payoffs[j]=Y
        controls[j]=Z
    
    Z_mean = np.mean(controls)

    
    d1 = (np.log(S0 / K) + (r_E + 0.5 * sigma_E**2) * T) / (sigma_E * np.sqrt(T))
    d2 = d1 - sigma_E * np.sqrt(T)
    Z_exact = np.exp(-r * T) * (S0 * np.exp(r_E * T) * approx_norm_cdf(d1) - K * approx_norm_cdf(d2))

    P_control = np.mean(payoffs) -  Z_mean + Z_exact

    # Calcul de l'erreur et de l'intervalle de confiance à 90%
    std = np.std(payoffs - controls)
    error = 1.65 * std / np.sqrt(M)

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

# P_VC = PDt_MC_control(T, K, S0, r, sigma, N, M)
# print(f"Prix estimé par Monte Carlo avec variable de contrôle : {P_VC[0]:.4f}")

# print(f"Intervalle de confiance à 90% : [{P_VC[1]:.4f}, {P_VC[2]:.4f}]")


#####################################
# Q8 : Tracés de P^Δt,MC et de P^Δt,MC,ctrl 
####################################


m = np.logspace(1, 4, 100, dtype=int)  # de 10 à 10_000, 100 points équitablement réparti en log

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

# vals = np.linspace(1,0, 50,endpoint=False) 
# K_vals = ss.beta(2,2).ppf(vals)*2           # liste de points dans (0,2] avec une concentration autour de 1
# # print(K_vals)
# plt.hist(K_vals,25)
# plt.show()

# Prices_MC_ctrl = []
# CI_up_ctrl = []
# CI_down_ctrl = []
# Prices_TW = []

# for K in K_vals:
#     price_ctrl, ci_down_ctrl, ci_up_ctrl, _ = PDt_MC_control(T, K, S0, r, sigma, N, M)
#     Prices_MC_ctrl.append(price_ctrl)
#     CI_up_ctrl.append(ci_up_ctrl)
#     CI_down_ctrl.append(ci_down_ctrl)
#     Prices_TW.append(PD_TW(T, K, S0, r, sigma, N))

# plt.close()  
# plt.figure(figsize=(10, 4)) 
# plt.plot(K_vals, Prices_MC_ctrl, label='P^Δt,MC,ctrl', linestyle='-')
# plt.fill_between(K_vals, CI_down_ctrl, CI_up_ctrl, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')
# plt.plot(K_vals, Prices_TW, label='P^Δt,TW', linestyle='--')
# plt.xlabel('Prix d\'exercice K')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique en fonction de K')
# plt.legend()
# plt.grid()
# plt.show()

# # Différence
# Prices_MC_ctrl = np.array(Prices_MC_ctrl)
# Prices_TW = np.array(Prices_TW)

# plt.close()
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, Prices_MC_ctrl - Prices_TW, label="P^Δt,MC,ctrl - P^Δt,TW", color="blue")
# plt.axhline(0, linestyle='--', color='gray')
# plt.xlabel("Prix d'exercice K")
# plt.ylabel("Différence")
# plt.title("Différence entre les deux estimateurs")

# plt.xlim(0.9, 1.1)  # Zoom sur l'axe des K autour de 1

# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


#####################################
# Q10 : Tracé de P^Δt,MC,ctrl en fonction de sigma
####################################

K = 1
# nb=50
# sigma_vals = np.linspace(0.8,0, nb,endpoint=False)
# Prices_MC_ctrl =np.zeros(nb)
# CI_up_ctrl =    np.zeros(nb)
# CI_down_ctrl =  np.zeros(nb)
# Prices_TW =     np.zeros(nb)

# for i in range(nb):
#     price_ctrl, ci_down_ctrl, ci_up_ctrl, _ = PDt_MC_control(T, K, S0, r, sigma_vals[i], N, M)
#     Prices_MC_ctrl[i]=price_ctrl
#     CI_up_ctrl[i] = ci_up_ctrl
#     CI_down_ctrl[i] = ci_down_ctrl
#     Prices_TW[i] = PD_TW(T, K, S0, r, sigma_vals[i], N)

# plt.close()  
# plt.figure(figsize=(10, 4))
# plt.plot(sigma_vals, Prices_MC_ctrl, label='P^Δt,MC,ctrl')
# plt.fill_between(sigma_vals, CI_down_ctrl, CI_up_ctrl, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')
# plt.plot(sigma_vals, Prices_TW, label='P^Δt,TW', linestyle='--')
# plt.xlabel('Volatilité σ')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique en fonction de σ')
# plt.legend()
# plt.grid()
# plt.show()

# # Différence
# diff = Prices_MC_ctrl - Prices_TW

# plt.close()  
# plt.figure(figsize=(10, 4)) 
# plt.plot(sigma_vals,diff, label="P^Δt,MC,ctrl - P^Δt,TW", color="blue")
# plt.axhline(0, linestyle='--', color='gray')
# plt.xlabel("Volatilité σ")
# plt.ylabel("Différence")
# plt.title("Différence entre les deux estimateurs")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#####################################
# Q11 : Tracé de P^Δt,TW en fonction K pour plusieurs Δt
####################################
# sigma = 0.3

# nb=100
# vals = np.linspace(1,0, nb,endpoint=False) 
# K_vals = ss.beta(2,2).ppf(vals)*2           # liste de points dans (0,2] avec une concentration autour de 1
# Prices_TW_0 =       np.zeros(nb)
# Prices_TW_1_252 =   np.zeros(nb)
# Prices_TW_1_52 =    np.zeros(nb)
# Prices_TW_1_12 =    np.zeros(nb)

# for k in range(nb):
#     Prices_TW_0[k]      = PD_TW(1, K_vals[k], S0, r, sigma, 1000000)  # Δt = 0
#     Prices_TW_1_252[k]  = PD_TW(1, K_vals[k], S0, r, sigma, 252)  # Δt = 1/252
#     Prices_TW_1_52[k]   = PD_TW(1, K_vals[k], S0, r, sigma, 52)  # Δt = 1/52
#     Prices_TW_1_12[k]   = PD_TW(1, K_vals[k], S0, r, sigma, 12)  # Δt = 1/12

# plt.close() 
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, Prices_TW_0, label='Δt = 0 (cas continu)', linestyle='-')
# plt.plot(K_vals, Prices_TW_1_252, label='Δt = 1/252 (un jour)', linestyle='-')
# plt.plot(K_vals, Prices_TW_1_52, label='Δt = 1/52 (une semaine)', linestyle='-')
# plt.plot(K_vals, Prices_TW_1_12, label='Δt = 1/12 (un mois)',linestyle='-')
# plt.xlabel('Prix d\'exercice K')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique selon Turnbull & Wakeman en fonction de Δt')
# plt.legend()
# plt.grid()

# plt.xlim(0.8, 1.2)
# plt.ylim(0, 0.2)

# # plt.tight_layout()
# # plt.show()

# plt.close() 
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, np.log10(Prices_TW_1_252  - Prices_TW_0), label='Δt = 1/252 (un jour)', linestyle='-')
# plt.plot(K_vals, np.log10(Prices_TW_1_52   - Prices_TW_0), label='Δt = 1/52 (une semaine)', linestyle='-')
# plt.plot(K_vals, np.log10(Prices_TW_1_12   - Prices_TW_0), label='Δt = 1/12 (un mois)',linestyle='-')
# plt.xlabel('log-différence de Prix d\'exercice K')
# plt.ylabel('Prix')
# plt.title('Prix de l\'option asiatique selon Turnbull & Wakeman en fonction de Δt')
# plt.legend()
# plt.grid()


# plt.tight_layout()
# plt.show()

#####################################
# Q12 : Tracé de la différence P^Δt,MC,ctrl - P^Δt,TW pour différents Δt
###################################

# nb=50
# vals = np.linspace(1,0, nb,endpoint=False) 
# K_vals = ss.beta(2,2).ppf(vals)*2           # liste de points dans (0,2] avec une concentration autour de 1
# Prices_diff_1_252 = []
# Prices_diff_1_52 = []
# Prices_diff_1_12 = []

# CI_up_diff_1_252 = []
# CI_down_diff_1_252 = []

# CI_up_diff_1_52 = []  
# CI_down_diff_1_52 = []  

# CI_up_diff_1_12 = []
# CI_down_diff_1_12 = []


# for K in K_vals:
#     a_252, b_252, c_252, _ = PDt_MC_control(1, K, S0, r, sigma, 252, M)  # Δt = 1/252
#     a_52, b_52, c_52, _ = PDt_MC_control(1, K, S0, r, sigma, 52, M)    # Δt = 1/52
#     a_12, b_12, c_12, _ = PDt_MC_control(1, K, S0, r, sigma, 12, M)    # Δt = 1/12
#     k_252 = PD_TW(1, K, S0, r, sigma, 252)
#     k_52 = PD_TW(1, K, S0, r, sigma, 52)
#     k_12 = PD_TW(1, K, S0, r, sigma, 12)

#     Prices_diff_1_252.append(a_252 - k_252 )  # Δt = 1/252
#     Prices_diff_1_52.append(a_52 - k_52)  # Δt = 1/52
#     Prices_diff_1_12.append(a_12 - k_12)  # Δt = 1/12

#     CI_up_diff_1_252.append(c_252 - k_252)  # Δt = 1/252
#     CI_down_diff_1_252.append(b_252 - k_252)  # Δt = 1/252

#     CI_up_diff_1_52.append(c_52 - k_52)  # Δt = 1/52
#     CI_down_diff_1_52.append(b_52 - k_52)  # Δt = 1/52

#     CI_up_diff_1_12.append(c_12 - k_12)  # Δt = 1/12
#     CI_down_diff_1_12.append(b_12 - k_12)  # Δt = 1/12

# plt.close() 
# plt.figure(figsize=(10, 4))
# plt.plot(K_vals, Prices_diff_1_252, label='Δt = 1/252 (un jour)')
# plt.plot(K_vals, Prices_diff_1_52, label='Δt = 1/52 (une semaine)')
# plt.plot(K_vals, Prices_diff_1_12, label='Δt = 1/12 (un mois)')

# plt.fill_between(K_vals, CI_down_diff_1_252, CI_up_diff_1_252, alpha=0.2)
# plt.fill_between(K_vals, CI_down_diff_1_52, CI_up_diff_1_52, alpha=0.2)
# plt.fill_between(K_vals, CI_down_diff_1_12, CI_up_diff_1_12, alpha=0.2)

# plt.xlabel('Prix d\'exercice K')
# plt.ylabel('Différence')
# plt.ylabel(r"$P^{\Delta t, MC, ctrl} - P^{\Delta t, TW}$")
# plt.axhline(0, linestyle='--', color='gray')
# plt.title(r'Différence entre $P^{\Delta t, MC, ctrl}$  et $P^{\Delta t, TW}$ en fonction de K pour plusieurs Δt')
# plt.legend()
# plt.grid()

# # plt.xlim(0.9, 1.1)   #zoom sur l'axe des K autour de 1
# # plt.ylim(-0.0006, 0.0002)

# plt.tight_layout()
# plt.show()


#####################################
# Q13 : temps de calcul pour différentes échelles de temps
# ####################################

# K=1
# S0=1
# r=0.01
# sigma =0.3
# M=1000
# dt = [12,52,252]
# elapse_control, elapse_TW = np.zeros(len(dt)), np.zeros(len(dt))
# nb=100   #nombre de temps échantillons

# for i in range(len(dt)):
#     start = time.time()

#     for n in range(nb):
#         lorem = PDt_MC_control(1, K, S0, r, sigma, dt[i], M)
#     elapse_control[i] = time.time()-start
#     elapse_control[i] = round(elapse_control[i]/nb,5)
    
#     start2 = time.time()
#     for n in range(nb):
#         lorem = PD_TW(1, K, S0, r, sigma, 12)
#     elapse_TW[i] = time.time()- start2
#     elapse_TW[i] = round(elapse_TW[i]/nb,8)

#     print("pour l'écart ",dt[i])
#     print("l'estimateur de MC a donné un résultat en ", elapse_control[i])
#     print("et l'estimateur de TW a donné un résultat en ", elapse_TW[i])

# abs = ["dt=1/12","dt=1/52","dt=1/252"]

# # plt.bar(abs,elapse_control,label="temps de calcul de l'estimateur MC")
# # plt.bar(abs, elapse_TW, label="temps de calcul de l'estimateur TW")

# # plt.legend()
# # plt.show()


# temps_calc = {
#     'TW' : elapse_TW,
#     'MC' : elapse_control
# }

# temps_calc_log = {
#     'TW' : np.log(elapse_TW)/np.log(10),
#     'MC' : np.log(elapse_control)/np.log(10)
# }

# x = np.arange(len(abs))  # the label locations
# width = 0.25  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained')

# for attribute, measurement in temps_calc.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('temps de calcul (s)')
# ax.set_title('temps de calcul des deux estimateurs pour différents pas de temps ')
# ax.set_xticks(x + width, abs)
# ax.legend(loc='upper left', ncols=3)

# plt.savefig("time calc")

# plt.close()
# fig, ax = plt.subplots(layout='constrained')

# for attribute, measurement in temps_calc_log.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('log s')
# ax.set_title('temps de calcul des deux estimateurs pour différents pas de temps')
# ax.set_xticks(x + width, abs)
# ax.legend(loc='upper left', ncols=3)

# plt.savefig("log-time calc")




#####################################
# Q16 : Réduction de variance par variable de contrôle en fonction de K
# ####################################


def PDt_MC_control2(T, K, S0, r, sigma, N, M):

    dt = T / N
    t = np.linspace(0, T, N+1)

    puts = np.zeros(M)


    for j in range(M):
        W = Brown_traj(T, N)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        
        S_bar = np.mean(S[1:])
        puts[j] = np.exp(-r * T) * max(K - S_bar, 0)
        
    control_term = (S0 / N) * np.sum(np.exp(r * np.arange(1, N + 1) * dt)) - K
    control_price = np.exp(-r * T) * control_term

    P = control_price + np.mean(puts)

    # Erreur-type
    std = np.std(puts)
    error = 1.65 * std / np.sqrt(M)

    CI_up = P + error
    CI_down = P - error

    return P, CI_down, CI_up, error


# Test d'application :
T = 0.5
S0 = 1
r = 0.01
sigma = 0.3
dt = 1 / 252
N = int(T / dt)  
M = 10000 


vals = np.linspace(1,0, 50,endpoint=False) 
K_vals = ss.beta(2,2).ppf(vals)*2           # liste de points dans (0,2] avec une concentration autour de 1

Prices_MC_ctrl = []
CI_up_ctrl = []
CI_down_ctrl = []
Prices_MC_ctrl2 = []
CI_up_ctrl2 = []
CI_down_ctrl2 = []
Prices_TW = []

for K in K_vals:
    price_ctrl, ci_down_ctrl, ci_up_ctrl, _ = PDt_MC_control(T, K, S0, r, sigma, N, M)
    price_ctrl2, ci_down_ctrl2, ci_up_ctrl2, _ = PDt_MC_control2(T, K, S0, r, sigma, N, M)
    Prices_MC_ctrl.append(price_ctrl)
    CI_up_ctrl.append(ci_up_ctrl)
    CI_down_ctrl.append(ci_down_ctrl)
    Prices_MC_ctrl2.append(price_ctrl2)
    CI_up_ctrl2.append(ci_up_ctrl2)
    CI_down_ctrl2.append(ci_down_ctrl2)
    Prices_TW.append(PD_TW(T, K, S0, r, sigma, N))

plt.close()  
plt.figure(figsize=(10, 4)) 
plt.plot(K_vals, Prices_MC_ctrl, label='P^Δt,MC,ctrl', linestyle='-')
plt.fill_between(K_vals, CI_down_ctrl, CI_up_ctrl, alpha=0.2, label='IC 90% P^Δt,MC,ctrl')
plt.plot(K_vals, Prices_MC_ctrl2, label='P^Δt,MC,ctrl2', linestyle='-')
plt.fill_between(K_vals, CI_down_ctrl2, CI_up_ctrl2, alpha=0.2, label='IC 90% P^Δt,MC,ctrl2')
plt.plot(K_vals, Prices_TW, label='P^Δt,TW', linestyle='--')
plt.xlabel('Strike K')
plt.ylabel('Prix')
plt.title('Prix de l\'option asiatique en fonction de K')
plt.legend()
plt.grid()
plt.show()