#------------------------------------------------------------------------------
#
# GLASMA EVOLUTION -- FULL HOTSPOT MODEL
#
#------------------------------------------------------------------------------
#
# Il codice ha inizializzazioni per pp, pA e AA. 
# Sia per p che per A, assume un modello a hotspots.
# Il numero di partecipanti (Npart) sui due proiettili è definito dall'utente.
# Sono associati 3 hotspots per partecipante.
# Le medie di epsilon, PL, PT e altre sono pesate con la distribuzione
# di densità di energia, che è calcolata durante la soluzione delle CYM.
# Il codice calcola anche eps2, eps3, evento per evento
#
# Il main lavora insieme al parallel_main_V1.py che parallelizza sugli eventi.
# 
#------------------------------------------------------------------------------
import sys
import warnings
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # forza backend non-GUI (salva su file invece di mostrare a schermo)


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, polar, svd
import time 


import datetime
import os
import csv
# from mpl_toolkits.mplot3d import Axes3D  # in alcune versioni di matplotlib serve
import glasma_config as cfg


from numpy.linalg import solve, norm
# from scipy.fft import fft2, ifft2

try:
    from scipy.fft import fft2, ifft2
    print("✔ Using scipy.fft backend")
except ImportError:
    from numpy.fft import fft2, ifft2
    print("⚠️ scipy.fft not available, using numpy.fft backend")


# =============================================================================
# PARAMETRI GLOBALI da glasma.config.py
# =============================================================================

tau_start = cfg.tau_start
tf = cfg.tf
n_steps = cfg.n_steps

step_interval = cfg.step_interval   # misura l'intervallo per gli step di scrittura in CYM evolution
step_target = cfg.step_target    # step temporale per calcolare quantità 3D

Npart_p = cfg.Npart_p     # numero di partecipanti nel proiettile p (per protone Npart_p=1)
Npart_A = cfg.Npart_A     # numero di partecipanti nel proiettile A (per nucleo vedi tabella)

i_fluct = cfg.i_fluct                  # modello per la fluctuation strength
Delta = cfg.Delta                      # parametro libero Delta proporzionale alla fluctuation strength
tau_fluct_start = cfg.tau_fluct_start  # tempo a cui si attacca la fluttuazione
i_variance = cfg.i_variance            # modello per generazione fluttuazioni sul piano trasverso
lambdas= cfg.lambdas                   # correlation length esponenziale (i_fluct=4)


Nsheets = cfg.Nsheets    # NUMBER of SHEETS

## Parametri del reticolo trasverso
Lx = cfg.Lx
Ly = cfg.Ly   # Dimensioni fisiche in fm
Nx = cfg.Nx     # Numero di punti nel piano trasverso
Ny = cfg.Ny


## Parametri per l'asse di rapidità η
Leta = cfg.Leta
N_eta = cfg.N_eta              # Numero di punti in η (1 per boost-invarianza)


## IR regulator 
mIR = cfg.mIR  # GeV


#-------------------------------------------------------
##    CONTROLLI VARI
i_plot = cfg.i_plot        

i_plot_finale = cfg.i_plot_finale

     

i_filter = cfg.i_filter    
              
i_homogen_A = cfg.i_homogen_A  
i_homogen_p = cfg.i_homogen_p

iter_OR = cfg.ITER_OR 

 

NHQ = cfg.NHQ
plot_hqs = cfg.plot_hq

init_mom_space = cfg.init_mom_space
hq_flavor = cfg.hq_flavor
mass_HQ = cfg.mass_HQ
## Costanti fisiche
hbarc = 0.1973269804  # GeV·fm

mass_HQ =  mass_HQ / hbarc

tau_form = 1 / ( 2 * mass_HQ)
#tau_form = 0.02 #fm/c

#-------------------------------------------------------

#==============================================================================
#==============================================================================


Nhotspots_p = 3 * Npart_p
Nhotspots_A = 3 * Npart_A

t0 = tau_start
dt = (tf-t0)/n_steps


a_perp = Lx / Nx           # Passo reticolare in fm
a_eta = Leta/N_eta           # Passo reticolare in η

#==============================================================================
#==============================================================================


## Parametri QCD
Nc = 3                 # Numero di colori
alpha_s = 0.3          # Costante di accoppiamento forte
g = np.sqrt(4 * np.pi * alpha_s)  # Costante di accoppiamento di QCD
g = 2

mIRfm=mIR / hbarc     # IR regulator in 1/fm

#==============================================================================
#==============================================================================


## Coordinate fisiche centrate
x = (np.arange(Nx) - Nx//2) * a_perp
y = (np.arange(Ny) - Ny//2) * a_perp
X, Y = np.meshgrid(x, y, indexing='ij')
## Coordinate fisiche centrata in η
eta_phys = (np.arange(N_eta) - N_eta//2) * a_eta
neta_zero = np.argmin(np.abs(eta_phys))  # trova l'indice con |eta_phys| minimo

#==============================================================================
#==============================================================================

## Parametri del nucleo (A)
mu_A = cfg.mu_A             # Valore tipico di μ per il nucleo in GeV
mu_A_fm = mu_A / hbarc  # Valore tipico di μ per il nucleo in 1/fm

## Parametri per le Tp
B_cq = 4.0 * hbarc**2  # fm²
B_q = 0.3 * hbarc**2   # fm²
# B_q = 0.9 * hbarc**2   # fm²
c = 1.25                # Fattore proporzionale tra Q_s e gμ
xg = 3.94         # Valore efficace per  xg(x, Q₀²) 

#==============================================================================
#==============================================================================

#------------------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------------------

if len(sys.argv) == 3:
    nrun = int(sys.argv[1])
    NRUNTOT = int(sys.argv[2])
else:
    print("⚠️ Nessun argomento passato, uso nrun=0, NRUNTOT=1 per test/profiling.")
    nrun = 0
    NRUNTOT = 1


# nrun = int(sys.argv[1])

def get_better_seed(nrun=None):
    """
    Seed più robusto, combina:
    - microsecondi attuali
    - PID del processo
    - indice del run (opzionale)
    """
    now = datetime.datetime.now()
    time_part = int(now.strftime("%H%M%S%f"))   
    pid = os.getpid()
    seed = time_part + pid
    if nrun is not None:
        seed += int(nrun)
    return seed % (2**32)





def generate_Tp(NHS):
    Tp = np.zeros_like(X)
    quark_positions = []
    for _ in range(NHS):
        x_i = np.random.normal(0.0, np.sqrt(B_cq))
        y_i = np.random.normal(0.0, np.sqrt(B_cq))
        quark_positions.append((x_i, y_i))
        Tp += np.exp(-((X - x_i)**2 + (Y - y_i)**2) / (2 * B_q))
    Tp *= 1.0 / (NHS * 2 * np.pi * B_q)
    return Tp, quark_positions



def compute_Qs_from_Tp(Tp):
    """
    Calcola Qs(x, y) a partire da Tp(x, y).

    Usa:
    - Nc, alpha_s, xg (valori globali)

    Ritorna:
    - Qs(x, y): array 2D con la scala di saturazione in ogni punto [GeV]
    """
    prefactor = (2 * np.pi**2 / Nc) * alpha_s * xg
    Qs2 = prefactor * Tp
    Qs = np.sqrt(Qs2)        # in 1/fm
    return Qs

def compute_mu_from_Qs(Qs,i_proie):
    """
    Calcola μ(x, y) dal Qs(x, y) secondo:
        μ = (c / g^2) * Qs

    Usa:
    - c: parametro empirico
    - g: costante di accoppiamento (globale)

    Ritorna:
    - mu(x, y): array 2D
    """
    # ————— param gaussian quick-knob —————
    sigma_mu = B_cq          # [fm^2] 
    GW = np.exp(-(X**2 + Y**2) / (2 * sigma_mu))      # gaussiana centrata in (0,0)
    # ————————————————————————————————
    
    if i_proie == 1:  # Se è un protone
        if i_homogen_p == 1:
            base = np.full_like(Qs, fill_value=mu_A_fm)  # mu omogeneo sul piano trasverso 
            mu = base
        elif i_homogen_p == 2:
            base = np.full_like(Qs, fill_value=mu_A_fm)  # tubo circolare gaussiano 
            mu = GW * base    
        else:     
            mu = (c / g**2) * Qs # mu segue la Tp
    elif i_proie == 2:
        if i_homogen_A == 1:
            base = np.full_like(Qs, fill_value=mu_A_fm)  # mu omogeneo sul piano trasverso 
            mu = base
        elif i_homogen_A == 2:
            base = np.full_like(Qs, fill_value=mu_A_fm)  # tubo circolare gaussiano 
            mu = GW * base    
        else:     
            mu = (c / g**2) * Qs # mu segue la Tp
    return mu    # in 1/fm


def generate_rho_field(gmu, Nc=3):
    """
    Genera le cariche di colore rho^a(x, y) con varianza gmu(x, y)^2.

    Parametri:
    - gmu: array 2D con gμ(x, y) in fm⁻¹
    - Nc: numero di colori (default 3 → 8 generatori)

    Ritorna:
    - rho: array shape (Nc²-1, Nx, Ny)
    """
    Nc2m1 = Nc**2 - 1
    Nx, Ny = gmu.shape
    rho_raw = np.random.normal(0.0, 1.0, size=(Nc2m1, Nx, Ny))

     
    
    # Moltiplica per gmu(x, y) → broadcasting
    rho_core = rho_raw * gmu[None, :, :]  / (a_perp * np.sqrt(Nsheets))

    # Crea array con indice 0 vuoto (zero)
    rho = np.zeros((Nc2m1 + 1, Nx, Ny))
    rho[1:] = rho_core

    return rho


def solve_poisson_fft_comme(rho, m, a_perp):     # fa le fft con np
    """
    Risolve l'equazione di Poisson per ogni componente colore a:
        Δ⊥ α^a(x, y) = -ρ^a(x, y)
    via trasformata di Fourier.

    Parametri:
    - rho: array di shape (Nc², Nx, Ny), con rho[0] vuoto (non usato)
    - m: regolatore IR in fm⁻¹
    - a_perp: passo reticolare in fm

    Ritorna:
    - alpha: array di shape (Nc², Nx, Ny)
    """
    
    
    
    
    Nc2 = rho.shape[0]
    Nx, Ny = rho.shape[1:]

    # Costruisci la griglia dei momenti discreti
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=a_perp)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=a_perp)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Costruisci k_perp^2 discreto
    ktilde2 = (2 / a_perp)**2 * (
        np.sin(KX * a_perp / 2)**2 +
        np.sin(KY * a_perp / 2)**2
    )
    
    # ktilde2=KX**2 + KY**2

    alpha = np.zeros_like(rho, dtype=np.float64)

    # Loop su tutte le componenti a (ignorando rho[0])
    for a in range(1, Nc2):
        rho_k = np.fft.fft2(rho[a], norm="ortho")
        alpha_k = rho_k / (ktilde2 + m**2)
        alpha[a] = np.fft.ifft2(alpha_k, norm="ortho").real
        
        
        
        
    

    return alpha






def solve_poisson_fft(rho, m, a_perp):   # fa le fft con scipy
    """
    Risolve Δ⊥ α^a(x, y) = -ρ^a(x, y) via FFT, usando pyFFTW.
    
    Parametri:
    - rho: array di shape (Nc², Nx, Ny), con rho[0] vuoto (non usato)
    - m: massa IR (in fm⁻¹)
    - a_perp: passo reticolare in fm
    
    Ritorna:
    - alpha: array di shape (Nc², Nx, Ny)
    """

     


    Nc2 = rho.shape[0]
    Nx, Ny = rho.shape[1:]

    # === Costruzione della griglia dei momenti discreti
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=a_perp)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=a_perp)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    ktilde2 = (2 / a_perp)**2 * (
        np.sin(KX * a_perp / 2)**2 +
        np.sin(KY * a_perp / 2)**2
    )

    # === Array per il risultato
    alpha = np.zeros_like(rho, dtype=np.float64)

    # === Trasformate vettorializzate su tutte le componenti a > 0
    rho_k = fft2(rho[1:], norm="ortho", axes=(-2, -1))

    alpha_k = rho_k / (ktilde2 + m**2)[np.newaxis, :, :]
    
    alpha[1:] = ifft2(alpha_k, norm="ortho", axes=(-2, -1)).real

    return alpha


 

def get_su3_generators():
    """
    Restituisce gli 8 generatori t^a = λ^a / 2 come array (8, 3, 3)
    """
    from numpy import array
    
    
    fatt8 = 1/(np.sqrt(3))
    
    # Matrici di Gell-Mann λ^a
    lambda_matrices = [
        array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
        array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
        array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
        fatt8 *array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) 
    ]
    return np.array(lambda_matrices) / 2  # t^a = λ^a / 2


def expm_su3_via(M):
    """
    M: array di shape (...,3,3), dtype complex128, dato che M = i*A con A 
    Hermitiana traceless.
    Ritorna expm(M) = exp(i*A)
    """
    # 1) ricavo A Hermitiana
    A = -1j * M

    # 2) diagonalizzo A con eigh
    w, U = np.linalg.eigh(A)           # w: (...,3) reali, U: (...,3,3) unitario

    # 3) exp(i*lambda) == exp(w)
    expw = np.exp(1j * w)                   # shape (...,3)

    # 4) ricostruisco expm(M) = U @ diag(expw) @ U^H
    return (U * expw[..., None, :]) @ U.conj().swapaxes(-1, -2)

def compute_wilson_lines_111(alpha, g, t_a):
    """
    Calcola le matrici di Wilson V(x, y) = exp[i g alpha^a(x, y) t^a] su ogni punto della griglia.

    Parametri:
    - alpha: array shape (Nc², Nx, Ny), con alpha[0] vuoto (padding)
    - g: costante di accoppiamento (es. sqrt(4π * α_s))
    - t_a: array shape (8, 3, 3), generatori di SU(3), t^a = λ^a / 2

    Ritorna:
    - V: array shape (Nx, Ny, 3, 3), con matrici di Wilson unitarie SU(3)
    """
    Nc2, Nx, Ny = alpha.shape
    assert Nc2 == 9, "alpha deve includere padding alpha[0], totale Nc² componenti (0 incluso)"

    V = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)  # Matrice di Wilson 3x3

    for i in range(Nx):
        for j in range(Ny):
            M = np.zeros((3, 3), dtype=np.complex128)
            # Somma dei termini i g alpha^a t^a per ogni a
            for a in range(1, Nc2):  # Salta alpha[0]
                M += 1j * g * alpha[a, i, j] * t_a[a - 1]  # t_a ha 8 generatori → a-1
            # Calcola l'esponenziale della matrice M per ottenere la matrice di Wilson
            # V[i, j] = expm(M)
            V[i, j] = expm_su3_via(M)

    return V

def expm_su3_batch(M):
    """
    Calcola exp(M) = U @ diag(exp(i * lambda)) @ U^†
    per batch di M Hermitiane (via M = i A).
    M: shape (Nx, Ny, 3, 3)
    """
    A = -1j * M
    # eigensystem batch: U unitario, w reali
    w, U = np.linalg.eigh(A)  # shape: (Nx, Ny, 3), (Nx, Ny, 3, 3)
    expw = np.exp(1j * w)     # (Nx, Ny, 3)

    # Ricostruzione: expM = U @ diag(expw) @ U†
    U_expw = U * expw[..., None, :]      # broadcasting (Nx,Ny,3,3)
    expM = U_expw @ np.conj(np.swapaxes(U, -1, -2))
    return expM

def compute_wilson_lines(alpha, g, t_a):
    """
    Vettorizzata: costruisce M[i,j] = ig α^a t^a e poi calcola exp(M) via diagonalizzazione.
    """
    Nc2, Nx, Ny = alpha.shape
    assert Nc2 == 9

    # Costruisci M[i,j] = somma_a i g alpha[a, i, j] * t_a[a-1]
    M = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)

    for a in range(1, Nc2):
        M += 1j * g * alpha[a]  [:, :, None, None] * t_a[a-1][None, None, :, :]

    # Calcola expm(M) in maniera vettoriale
    V = expm_su3_batch(M)
    return V



def compute_gauge_link_x_periodic(V_total):
    """
    Calcola il gauge link nella direzione x per ogni punto della griglia
    con condizioni al contorno periodiche.
    """
    Nx, Ny, _, _ = V_total.shape
    U_x = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)  # Array per i gauge links nella direzione x

    for i in range(Nx):  # Considera tutte le x, con la periodicità
        for j in range(Ny):  # Considera tutte le y
            # Se i = Nx-1, usa il primo punto (periodicità)
            next_i = (i + 1) % Nx  # Uso modulo per garantire periodicità
            U_x[i, j] = np.dot(V_total[i, j], np.transpose(np.conj(V_total[next_i, j])))

    return U_x

def compute_gauge_link_y_periodic(V_total):
    """
    Calcola il gauge link nella direzione y per ogni punto della griglia
    con condizioni al contorno periodiche.
    """
    Nx, Ny, _, _ = V_total.shape
    U_y = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)  # Array per i gauge links nella direzione y

    for i in range(Nx):  # Considera tutte le x
        for j in range(Ny):  # Considera tutte le y
            # Se j = Ny-1, usa il primo punto (periodicità)
            next_j = (j + 1) % Ny  # Uso modulo per garantire periodicità
            U_y[i, j] = np.dot(V_total[i, j], np.transpose(np.conj(V_total[i, next_j])))

    return U_y


def compute_gauge_links(i_proiettile):
    """
    Calcola i gauge links U_x e U_y su tutta la griglia per Nsheets.
    
    Args:
        V_total (ndarray): matrice totale delle Wilson lines per ogni punto della griglia (Nx, Ny, 3, 3).
        Nx (int): numero di punti sulla griglia lungo x.
        Ny (int): numero di punti sulla griglia lungo y.
        Nsheets (int): numero di fogli.
        g (float): costante di accoppiamento.
        t_a (ndarray): matrici generatrici di Lie algebra (Nc x Nc).
        hbarc (float): costante hbarc.
        mIRfm (float): massa in unità di fm.
        a_perp (float): passo del reticolo.
    
    Returns:
        U_x (ndarray): gauge link lungo la direzione x.
        U_y (ndarray): gauge link lungo la direzione y.
    """
    # Ottieni i generatori SU(3)
    t_a = get_su3_generators()
    
    # Inizializza V_total come matrice identità 3x3 per ogni punto della griglia
    V_total = np.tile(np.eye(3, dtype=np.complex128), (Nx, Ny, 1, 1))
    
   



    if i_proiettile == 1:
        Tp, quark_positions = generate_Tp(Nhotspots_p)
        Tp = Npart_p * Tp   # La Tp nel proiettile 2 va normalizzata a Npart
        export_single_Tp(x, y, Tp, i_proiettile)
    elif i_proiettile == 2:    
        Tp, quark_positions = generate_Tp(Nhotspots_A)
        Tp = Npart_A * Tp   # La Tp nel proiettile 2 va normalizzata a Npart
        export_single_Tp(x, y, Tp, i_proiettile)
        
    
    Qs = compute_Qs_from_Tp(Tp)  
    gmu = g * compute_mu_from_Qs(Qs,i_proiettile)
    gmu_GeV = gmu * hbarc


    gmu_max = np.max(gmu_GeV)
    gmu_norm = gmu_GeV / gmu_max
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    gmu_flat = gmu_norm.flatten()
    data = np.column_stack((X_flat, Y_flat, gmu_flat))
    filename = f"HQinitializations/hq_init_event{nrun}_proj{i_proiettile}.csv"
    np.savetxt(filename, data, delimiter=",", fmt="%.6e")    


    seedA = get_time_seed()
    np.random.seed(seedA)
    
    
    


    
    # Ciclo per calcolare le Wilson lines e aggiornare V_total
    for ns in range(1, Nsheets + 1):
        print(f"i_proiettile={i_proiettile}, ns={ns}")

        
        # 2. Calcola Qs
        # Qs = compute_Qs_from_Tp(Tp)
        # gmu = g * compute_mu_from_Qs(Qs,i_proiettile)
        # gmu_GeV = gmu * hbarc
        # # 4. Calcola rho
        # seedA = get_time_seed()
        # np.random.seed(seedA)

        rho = generate_rho_field(gmu, Nc=3)
        # time.sleep(0.15)
        
        
        # print_rho_stats(rho, a=1)
        
        # 5. Calcola alpha
        
        
        
        alpha = solve_poisson_fft(rho, mIRfm, a_perp)
            # print("Statistiche di alpha^a:")
            # print(f"  Max |alpha|: {np.max(np.abs(alpha[1:]))}")  # Salta alpha[0]
            # print(f"  Media |alpha|: {np.mean(np.abs(alpha[1:]))}")
            # print(f"  Deviazione std: {np.std(alpha[1:])}")
        
            
        
        
        
        
        
        # 6. Calcola la matrice di Wilson per questo foglio
        Vns = compute_wilson_lines(alpha, g, t_a)
        # print("Statistiche di V - I:")
        # print(f"  Max |V - I|: {np.max(np.abs(Vns - np.eye(3)))}")
        # print(f"  Media |V - I|: {np.mean(np.abs(Vns - np.eye(3)))}")
        
        # 7. Aggiorna V_total moltiplicando progressivamente
        for i in range(Nx):
            for j in range(Ny):
                # Moltiplica la matrice V_total con la matrice Vns per ogni punto (i, j) nella griglia
                V_total[i, j] = np.dot(V_total[i, j], Vns[i, j])  # Prodotto matrice progressivo



        if i_plot == 1 and ns == 1:
            # plot_quantity(X, Y, Tp, 
            #               title=f"Densità trasversa {i_proiettile} $T_p(x, y)$",  
            #               label=r"$T_p(x, y)$", 
            #               hotspots=quark_positions)
            
            # plot_quantity(X, Y, Qs_GeV, 
            #               title=f"$Q_s$ {i_proiettile} (GeV)",  
            #               label=r"$Q_s$ [GeV]")
            
            plot_quantity(X, Y, gmu_GeV, 
                          title=f"$g*mu$ {i_proiettile} (GeV)",  
                          label=r"$g*mu$ [GeV]")

        # print(gmu_GeV)
        # print(g)
        
    
     
    
    
    
    # Calcola i gauge links
    U_x = compute_gauge_link_x_periodic(V_total)
    U_y = compute_gauge_link_y_periodic(V_total)
    
    



      
    
    # plot_delta_V(V_total)
    
    return U_x, U_y


# Calcola A_i = (1/ig) log U_i (usando logaritmo matriciale)
def compute_A(U, g):
    Nx = U.shape[0]
    A = np.zeros_like(U, dtype=complex)
    
    for x in range(Nx):
        for y in range(Nx):
            # Calcola il logaritmo della matrice 3x3 in (x,y)
            A[x, y] = -(1 / (1j * g)) * logm(U[x, y])
    return A


def Eeta_in(Ux, Uy, UAx, UAy, UBx, UBy):
    """
    Calcola la componente eta del campo del glasma usando gauge links separati per x e y.
    
    Parametri:
    - Ux, Uy: array di shape (Nx, Nx, 3, 3) - Gauge links totali nelle direzioni x e y
    - UAx, UAy: array di shape (Nx, Nx, 3, 3) - Gauge links del nucleo A nelle direzioni x e y
    - UBx, UBy: array di shape (Nx, Nx, 3, 3) - Gauge links del nucleo B nelle direzioni x e y
    - Nx: int - Dimensione del reticolo
    - a_perp: float - Passo reticolare
    - g: float - Costante di accoppiamento
    
    Restituisce:
    - E_eta: array di shape (Nx, Nx, 3, 3) - Campo E_eta
    """

    # E^eta(tau0) = -i g [alpha_R^i, alpha_L^i]  (tesi), con proiettile 1 = R, proiettile 2 = L
    
    # Matrice identità 3x3
    I = np.eye(3, dtype=complex)
    
    # Pre-fattore
    preff = -1j / (4 * g * a_perp**2)

    
    # Inizializzazione del campo E_eta
    E_eta = np.zeros((Nx, Nx, 3, 3), dtype=complex)
    
    for x in range(Nx):
        for y in range(Nx):
            # --- Termine per la direzione x ---
            xm = (x - 1) % Nx  # Punto (x-1, y) con condizioni periodiche
            term_x = (Ux[x, y] - I) @ (UBx[x, y].conj().T - UAx[x, y].conj().T) + \
                     (Ux[xm, y].conj().T - I) @ (UBx[xm, y] - UAx[xm, y])
            
            # --- Termine per la direzione y ---
            ym = (y - 1) % Nx  # Punto (x, y-1) con condizioni periodiche
            term_y = (Uy[x, y] - I) @ (UBy[x, y].conj().T - UAy[x, y].conj().T) + \
                     (Uy[x, ym].conj().T - I) @ (UBy[x, ym] - UAy[x, ym] )
            
            # Somma dei termini e anti-hermitiano coniugato
            E_eta[x, y] = preff * (term_x + term_y - (term_x + term_y).conj().T)
            
    return E_eta


# ---------- utility ---------------------------------------------------------

 












# ----------------------------------------------------------------------
#  SU(3) utilities
# ----------------------------------------------------------------------
def su3_generators():
    """Gell-Mann T_a = λ_a / 2, normalizzati Tr[T_a T_b] = ½ δ_ab."""
    lam = []
    lam.append(np.array([[0,1,0],[1,0,0],[0,0,0]]))                 # λ1
    lam.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]]))              # λ2
    lam.append(np.array([[1,0,0],[0,-1,0],[0,0,0]]))                # λ3
    lam.append(np.array([[0,0,1],[0,0,0],[1,0,0]]))                 # λ4
    lam.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]]))              # λ5
    lam.append(np.array([[0,0,0],[0,0,1],[0,1,0]]))                 # λ6
    lam.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]]))              # λ7
    lam.append(np.array([[1/np.sqrt(3),0,0],[0,1/np.sqrt(3),0],[0,0,-2/np.sqrt(3)]]))  # λ8
    return np.stack([m*0.5 for m in lam])                            # shape (8,3,3)

Ta = su3_generators()      # broadcastable array (8,3,3)

def project_to_su3(M):
    """Proiezione polare via SVD → SU(3)."""
    U, _, Vh = svd(M, full_matrices=False)
    U = U @ Vh
    U /= np.linalg.det(U)**(1/3)
    return U
# ----------------------------------------------------------------------
#  Cautun iterative solver (senza SAM)
# ----------------------------------------------------------------------
def solve_link(Ua, Ub, *, tol=1e-10, max_iter=200):
    """
    Risolve l’equazione (3.21) della tesi per un singolo link:
        Tr[T_a (Ua+Ub)(1+U)] - h.c. = 0
    usando l’algoritmo iterativo 3.24–3.31 (α-scaling) di Cautun.
    Restituisce U, n_iter, residuo_finale.
    """
    Nc = Ua.shape[0]
    eye = np.eye(Nc, dtype=complex)

    # 1) guess iniziale U = Proj[Ua * Ub]  (eq. 3.24)
    U = project_to_su3(Ua @ Ub)

    for k in range(1, max_iter+1):
        # 2) calcola f_a  (eq. 3.25)
        M1 = (Ua + Ub) @ (eye + U.conj().T)    # (Ua+Ub)(1+U†)
        fa = np.imag(np.tensordot(Ta.conj(), M1, axes=([1,2],[0,1])))  # 8-vet

        res = np.sum(np.abs(fa))               # ∑|f_a|
        if res < tol:
            return U 

        # 3) costruisci la matrice   Mab = Im Tr[T_b T_a (Ua+Ub) U†]   (eq. 3.27)
        M2 = (Ua + Ub) @ U.conj().T
        Mab = np.empty((8,8))
        for b in range(8):
            Mab[b] = np.real(
                np.tensordot(Ta[b] @ Ta, M2, axes=([1,2],[0,1]))
            )

        xb = solve(Mab, fa)                   # risolve il sistema lineare

        # 4) aggiorna con α-scaling (eq. 3.31, soglia 0.5)
        alpha = 0.5 if res > 0.5 else 1.0
        H = (xb[:,None,None] * Ta).sum(axis=0)  # i x_a T_a  (senza i)
        U = expm(1j * alpha * H) @ U           # U_new

    # max_iter raggiunto → torna ultimo U
    return U


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ---------- main driver -----------------------------------------------------
def build_total_links(U_Ax, U_Ay, U_Bx, U_By):
    """
    Costruisce U_x, U_y con Gauss law esatta.
    Tutti gli array devono avere shape (Nx, Ny, Nc, Nc) e dtype=complex128.
    """
    Nx, Ny = U_Ax.shape[:2]
    # Nc = U_Ax.shape[-1]
    U_x = np.empty_like(U_Ax, dtype=complex)
    U_y = np.empty_like(U_Ay, dtype=complex)
    
    
   
    for ix in range(Nx):
        for iy in range(Ny):
            # direzione x
            Ua = U_Ax[ix, iy]
            Ub = U_Bx[ix, iy]
            U_x[ix, iy] = solve_link(Ua, Ub)
            # direzione y
            Ua = U_Ay[ix, iy]
            Ub = U_By[ix, iy]
            U_y[ix, iy] = solve_link(Ua, Ub)

    return U_x, U_y






def initialization():
    """
    Esegue l'inizializzazione del Glasma calcolando:
      - i gauge link UA e UB per i due proiettili
      - A_tot_x, A_tot_y
      - Utot_x, Utot_y
      - Il campo Eeta in base ai gauge link
    Non prende parametri (usa i valori globali).
    Restituisce: Utot_x, Utot_y, Eeta
    """
    # Calcola i gauge links per i due proiettili
    for iA in [1, 2]:
        U_x, U_y = compute_gauge_links(iA)

        if iA == 1:
            UA_x, UA_y = U_x, U_y
        else:
            UB_x, UB_y = U_x, U_y



    # Costruisce i gauge link 'totali'
    Utot_x, Utot_y = build_total_links(UA_x, UA_y, UB_x, UB_y)

    # Calcola E^eta
    Eeta = Eeta_in(Utot_x, Utot_y, UA_x, UA_y, UB_x, UB_y)
    
    # Gauge link in direzione eta: identità 3x3 in ogni punto
    Utot_eta = np.tile(np.eye(3, dtype=complex), (Nx, Ny, 1, 1)).reshape(Nx, Ny, 3, 3)

    # Campi trasversi inizialmente nulli
    Ex = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    Ey = np.zeros((Nx, Ny, 3, 3), dtype=complex)

    return Utot_x, Utot_y, Utot_eta, Ex, Ey, Eeta



# def save_matrix_configuration(matrix, filename, folder_path, varname):
#     path = os.path.join(folder_path, filename)
#     np.savetxt(path,
#                matrix.reshape(-1, 3),
#                fmt = "%.10e",
#                header = f"{varname} shape={matrix.shape} dtype={matrix.dtype}")

#     # with open(path, 'w') as f:
#     #     f.write(f"# {varname} shape={matrix.shape} dtype={matrix.dtype}\n")
#     #     for ix in range(matrix.shape[0]):
#     #         for iy in range(matrix.shape[1]):
#     #             np.savetxt(f, matrix[ix, iy], fmt="%.10e")
#     #             f.write("\n")  # riga vuota dopo ogni matrice 3x3

#     # == debugging ==
#     matrix_check = np.loadtxt(path, dtype = complex).reshape(64, 64, 3, 3)
#     print(f"Salvataggio in txt di {varname}: {np.allclose(matrix,matrix_check, atol = 1e-9)}")

#     # with open(path, 'r') as f:
#     #     lines = [line.strip() for line in f 
#     #              if line.strip() and not line.startswith('#')]
#     # data = np.array([complex(val) for line in lines 
#     #                  for val in line.split()])
#     # matrix_check = data.reshape(matrix.shape)
#     # print(f"Salvataggio in txt di {varname}: {np.allclose(matrix, matrix_check, atol=1e-9)}")










def calculate_Eeta_trace(Eeta):
    """
    Calcola la traccia di Eeta × Eeta in ogni punto del reticolo.
    
    Parametri:
    - Eeta: array di shape (Nx, Nx, 3, 3) - Campo Eeta calcolato precedentemente
    
    Restituisce:
    - Eeta_trace: array di shape (Nx, Nx) - Traccia di Eeta×Eeta in ogni punto
    """
    Nx = Eeta.shape[0]
    Eeta_trace = np.zeros((Nx, Nx), dtype=complex)
    
    for x in range(Nx):
        for y in range(Nx):
            # Prodotto matriciale Eeta × Eeta
            Eeta_squared = np.dot(Eeta[x,y], Eeta[x,y])
            # Calcolo della traccia (somma degli elementi diagonali)
            Eeta_trace[x,y] = np.trace(Eeta_squared)
    
    return Eeta_trace





def extend_in_eta(U_xy, N_eta):
    """
    Estende un campo boost-invariante (Nx, Ny, 3, 3) lungo la direzione eta
    in (Nx, Ny, N_eta, 3, 3), copiando il contenuto su ogni slice eta.

    Parametri:
    - U_xy: array (Nx, Ny, 3, 3)
    - N_eta: numero di slice in direzione eta

    Ritorna:
    - U_eta_extended: array (Nx, Ny, N_eta, 3, 3)
    """
    return np.repeat(U_xy[:, :, np.newaxis, :, :], N_eta, axis=2)


def compute_F_eta_fluctuations_model(N_eta, Delta, N_perp, a_eta, i_fluct, lambdas):
    """
    Calcola la forma della funzione F(eta) che modellizza le fluttuazioni del glasma 
    in base al modello che si vuole utilizzare.

    i_fluct:
    1) 1 armonica
    2) scomposizione in 3 coseni
    3) gaussian random number 

    Parametri:
    - N_eta: numero di punti sulla griglia in eta
    - Delta: parametro libero per la strength delle fluttuazioni
    - N_perp:
    - a_eta: spaziatura reticolo in eta
    - modes: insieme I dei modi interi positivi della scomposizione 1
    
    Ritorna:
    - F_eta, in base al modello scelto via i_fluct 
    """
    L_eta = a_eta*N_eta                         #length della griglia in eta

    if i_fluct == 1:        
        eta = np.arange(N_eta) * a_eta - L_eta / 2  #tutti i valori delle eta su griglia [-L/2,+L/2]

        F_eta = (Delta / N_perp) * np.cos(2 * np.pi * eta / L_eta)

        return F_eta
    
    elif i_fluct == 2:
        modes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        card_I = len(modes)
        eta = np.arange(N_eta) * a_eta - L_eta / 2  #tutti i valori delle eta su griglia [-L/2,+L/2]

        #calcola il coseno sulla griglia
        cos_matrix = np.cos(2 * np.pi * modes[np.newaxis, :] * eta[:, np.newaxis] / L_eta)

        prefactor = (Delta / (N_perp * card_I))
        #calcola la F sulla griglia di eta 
        F_eta = prefactor * np.sum(cos_matrix, axis=1)

        return F_eta

    elif i_fluct == 3:
        F_eta = np.random.normal(0.0, Delta / np.sqrt(L_eta), size=N_eta)
        b = 0.05
        #da completare con il filtro

        # 1) Calcolare la FFT
        F_tilde = np.fft.fft(F_eta)

        # 2) Definire le frequenze e applicare il filtro
        # np.fft.fftfreq calcola le frequenze "standard" f = k / (N * dx)
        freqs = np.fft.fftfreq(N_eta, d=a_eta)

        # Nel paper, la variabile \nu rappresenta la "pulsazione" (rad/lunghezza), quindi \nu = 2 * pi * f
        nu = 2 * np.pi * freqs

        # Calcolo del filtro: exp(-|\nu|b)
        filtro = np.exp(-np.abs(nu) * b)

        # Moltiplicazione elemento per elemento
        F_tilde_filtrata = F_tilde * filtro

        # 3) Calcolare la trasformata inversa
        # Usiamo ifft e prendiamo solo la parte reale (.real)
        # Matematicamente il risultato è reale, ma numericamente potrebbero rimanere minuscoli residui immaginari (es. 1e-16j)
        F_eta_filtrata = np.fft.ifft(F_tilde_filtrata).real
        return F_eta_filtrata

    elif i_fluct == 4:
        f = np.random.normal(0.0, 1 / np.sqrt(a_eta), size=N_eta)
        # 1) Calcolare la FFT
        f_tilde = np.fft.fft(f)

        # 2) Definire le frequenze e applicare il filtro
        # np.fft.fftfreq calcola le frequenze "standard" f = k / (N * dx)
        freqs = np.fft.fftfreq(N_eta, d=a_eta)
        h = 2 * np.pi * freqs

        # Calcolo del filtro ello spazio dei momenti
        W = Delta / ((1+(lambdas**2)*(h**2))**2)

        # Moltiplicazione elemento per elemento
        F_tilde = W * f_tilde

        # 3) Calcolare la trasformata inversa
        # Usiamo ifft e prendiamo solo la parte reale (.real)
        # Matematicamente il risultato è reale, ma numericamente potrebbero rimanere minuscoli residui immaginari (es. 1e-16j)
        F = np.fft.ifft(F_tilde).real
        return F
    
    

def delta_fluctuations_glasma(Nx, Ny, N_eta, a_perp, a_eta, Delta, i_fluct, Ux, Uy, i_variance, lambdas):
    """
    Genera le fluttuazioni per le componenti i ed eta del campo elettrico del glasma.

    Genera una random configuration csi_i(x_perp) con varianza 1 e dev std 1/a_perp;
    calcola le fluttuazioni inserendo la csi_i nella relazione con la F_eta computata 
    con apposita funzione creata in precedenza. 
    """
    if i_variance == 0:
        #genera le csi random, per ogni indice di colore gluonico
        csi_x_a = np.random.normal(0.0, 1.0/a_perp, size=(8, Nx, Ny))
        csi_y_a = np.random.normal(0.0, 1.0/a_perp, size=(8, Nx, Ny))
    
    elif i_variance == 1:
        # 1. Costruisci i nomi dei due file
        file1 = f"HQinitializations/hq_init_event{nrun}_proj1.csv"
        file2 = f"HQinitializations/hq_init_event{nrun}_proj2.csv"

        # 2. Leggi i due file CSV
        data1 = np.loadtxt(file1, delimiter=",")  # colonne: x y gmu_norm
        data2 = np.loadtxt(file2, delimiter=",")  # stesso formato

        # 3. Estrai colonne (x, y sono uguali nei due)
        Nx = len(np.unique(data1[:, 0]))
        Ny = len(np.unique(data1[:, 1]))

        gmu1 = data1[:, 2].reshape(Nx, Ny)  # shape (Nx, Ny)
        gmu2 = data2[:, 2].reshape(Nx, Ny)  # shape (Nx, Ny)

        # 4. Calcola la densità normalizzata
        density = gmu1 * gmu2
        # assert np.all(density >= 0), f"Density contiene valori negativi! Minimo: {np.min(density)}"
        density /= np.max(density)  # normalizza (sommata fa 1)

        # 5. calcola dev std 
        sigma = (np.sqrt(density) / a_perp)[np.newaxis, :, :]  # shape (1, Nx, Ny) --> serve per il broadcasting

        # 6. genera le fluttuazioni
        csi_x_a = np.random.normal(0.0, sigma, size=(8, Nx, Ny))
        csi_y_a = np.random.normal(0.0, sigma, size=(8, Nx, Ny))


    t_a = get_su3_generators()  # shape: (8, 3, 3)

    csi_x_mat = np.einsum('axy,abc->xybc', csi_x_a, t_a)  # struttura SU(3) di colore per le csi
    csi_y_mat = np.einsum('axy,abc->xybc', csi_y_a, t_a)  # (Nx, Ny, 3, 3)

    csi_x_mat_ext = extend_in_eta(csi_x_mat, N_eta)
    csi_y_mat_ext = extend_in_eta(csi_y_mat, N_eta)

    #calcola la strength F_eta per le fluttuazioni e la differenza per la derivata in eta
    F_eta = compute_F_eta_fluctuations_model(N_eta, Delta, Nx, a_eta, i_fluct, lambdas)
    dF = np.roll(F_eta, +1) - F_eta #np.roll funziona automaticamente con le bc periodiche

    # #DEBUG
    # print(f"a_eta = {a_eta:.4e}")
    # print(f"Max |F_eta| = {np.max(np.abs(F_eta)):.4e}")
    print(f"Max |dF| = {np.max(np.abs(dF)):.4e}")
    #print(f"Max |dF/a_eta| = {(np.max(np.abs(dF/a_eta)))*hbarc**2:.4e}")

    #shifta i gauge link e le csi per la derivata covariante backward nella fluttuazione per E_eta
    Ux_shift   = np.roll(Ux,   +1, axis=0)   # shape (Nx, Ny, Neta, 3, 3)
    csi_x_shift = np.roll(csi_x_mat_ext, +1, axis=0)   # shape (Nx, Ny, 3, 3)
    CSI_x = np.einsum('xynab,xynbc,xyncd->xynad', Ux_shift.conj().swapaxes(-1,-2), csi_x_shift, Ux_shift)                         
    D_B_x = CSI_x - csi_x_mat_ext   #derivata backward x

    Uy_shift   = np.roll(Uy,   +1, axis=1)   # shape (Nx, Ny, Neta, 3, 3)
    csi_y_shift = np.roll(csi_y_mat_ext, +1, axis=1)   # shape (Nx, Ny, Neta, 3, 3)
    CSI_y = np.einsum('xynab,xynbc,xyncd->xynad', Uy_shift.conj().swapaxes(-1,-2), csi_y_shift, Uy_shift)                         
    D_B_y = CSI_y - csi_y_mat_ext   #derivata backward y

    sum_D_B = D_B_x + D_B_y   # shape (Nx, Ny,Neta,  3, 3)


    #calcola le fluttuazioni
    #delta_E_x = (dF[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * csi_x_mat[:, :, np.newaxis, :, :]) / a_eta
    delta_E_x = (dF[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * csi_x_mat_ext) / a_eta
    #delta_E_y = (dF[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * csi_y_mat[:, :, np.newaxis, :, :]) / a_eta
    delta_E_y = (dF[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * csi_y_mat_ext) / a_eta
    delta_E_eta = (-1/a_perp) * F_eta[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * sum_D_B[:, :, :, :, :]
    #   
    #                                   x            y     eta       a           b               x  y eta  a b
    # Iden_Test = np.zeros((Nx, Ny, N_eta, 3, 3), dtype=complex)
    # Iden_Test[:] = np.eye(3)
    # delta_E_eta = (-1/a_perp) * (Delta/Nx) * (1/a_perp) * Iden_Test
    
    
    return delta_E_x, delta_E_y, delta_E_eta


#------------------------------------------------------------------------------
# EVOLUZIONE CYM
#------------------------------------------------------------------------------


def update_gauge_links(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau, dtau):
    """
    Evoluzione dei gauge links in 3D (con η) usando il leapfrog.

    Parametri:
    - Ux, Uy, Ueta: gauge links (Nx, Ny, N_eta, 3, 3)
    - Ex, Ey, Eeta: campi elettrici (Nx, Ny, N_eta, 3, 3)
    - g: costante di accoppiamento
    - a_perp, a_eta: spacing reticolari
    - tau: tempo attuale
    - dtau: passo temporale

    Ritorna:
    - Ux_new, Uy_new, Ueta_new: gauge links aggiornati
    """
    Nx, Ny, Neta = Ux.shape[:3]
    tau_half = tau + dtau / 2

    Ux_new = np.zeros_like(Ux, dtype=complex)
    Uy_new = np.zeros_like(Uy, dtype=complex)
    Ueta_new = np.zeros_like(Ueta, dtype=complex)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Neta):
                # X
                arg_x = -1j * dtau * g * a_perp * Ex[i, j, k] / tau_half
                # Ux_new[i, j, k] = expm(arg_x) @ Ux[i, j, k]
                Ux_new[i, j, k] = expm_su3_via(arg_x) @ Ux[i, j, k]

                # Y
                arg_y = -1j * dtau * g * a_perp * Ey[i, j, k] / tau_half
                # Uy_new[i, j, k] = expm(arg_y) @ Uy[i, j, k]
                Uy_new[i, j, k] = expm_su3_via(arg_y) @ Uy[i, j, k]

                # ETA
                arg_eta = -1j * dtau * g * a_eta * tau_half * Eeta[i, j, k]
                # Ueta_new[i, j, k] = expm(arg_eta) @ Ueta[i, j, k]
                Ueta_new[i, j, k] = expm_su3_via(arg_eta) @ Ueta[i, j, k]

    return Ux_new, Uy_new, Ueta_new


def update_electric_fields(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau, dtau,
                            log_step=False, include_spatial_terms=True, include_eta_terms=True):
    Ex_new = Ex.copy()
    Ey_new = Ey.copy()
    Eeta_new = Eeta.copy()
    
    Neta = N_eta

    # --- E^x ---
    coeff_eta = (1j * dtau) / (2 * g * a_eta**2 * a_perp * tau)
    coeff_yx = (1j * dtau * tau) / (2 * g * a_perp**3)
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                etam = (eta - 1) % Neta
    
                # --- Longitudinal contribution [U_{ηx} + U_{-ηx} - h.c.] ---
                # U_{ηx}(x)
                U1 = Ueta[x, y, eta]
                U2 = Ux[x, y, etap]
                U3 = np.conj(Ueta[xp, y, eta].T)
                U4 = np.conj(Ux[x, y, eta].T)
                U_eta_x = U1 @ U2 @ U3 @ U4
    
                # U_{-ηx}(x)
                U1m = Ueta[x, y, etam].conj().T
                U2m = Ux[x, y, etam]
                U3m = Ueta[xp, y, etam]
                U4m = np.conj(Ux[x, y, eta].T)
                U_m_eta_x = U1m @ U2m @ U3m @ U4m
    
                comm_eta = (U_eta_x + U_m_eta_x) - (U_eta_x + U_m_eta_x).conj().T
    
                # --- Transverse contribution [U_{yx} + U_{-yx} - h.c.] ---
                # U_{yx}(x)
                U1 = Uy[x, y, eta]
                U2 = Ux[x, yp, eta]
                U3 = np.conj(Uy[xp, y, eta].T)
                U4 = np.conj(Ux[x, y, eta].T)
                U_y_x = U1 @ U2 @ U3 @ U4
    
                # U_{-yx}(x)
                U1m = Uy[x, ym, eta].conj().T
                U2m = Ux[x, ym, eta]
                U3m = Uy[xp, ym, eta]
                U4m = np.conj(Ux[x, y, eta].T)
                U_m_y_x = U1m @ U2m @ U3m @ U4m
    
                comm_yx = (U_y_x + U_m_y_x) - (U_y_x + U_m_y_x).conj().T
    
                # --- Aggiorna Ex ---
                Ex_new[x, y, eta] += coeff_eta * comm_eta + coeff_yx * comm_yx


    # --- E^y ---
    coeff_eta = (1j * dtau) / (2 * g * a_eta**2 * a_perp * tau)
    coeff_yx = (1j * dtau * tau) / (2 * g * a_perp**3)
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                etam = (eta - 1) % Neta
    
                # --- Longitudinal contribution [U_{ηx} + U_{-ηx} - h.c.] ---
                # U_{ηy}(x)
                U1 = Ueta[x, y, eta]
                U2 = Uy[x, y, etap]
                U3 = np.conj(Ueta[x, yp, eta].T)
                U4 = np.conj(Uy[x, y, eta].T)
                U_eta_y = U1 @ U2 @ U3 @ U4
    
                # U_{-ηy}(x)
                U1m = Ueta[x, y, etam].conj().T
                U2m = Uy[x, y, etam]
                U3m = Ueta[x, yp, etam]
                U4m = np.conj(Uy[x, y, eta].T)
                U_m_eta_y = U1m @ U2m @ U3m @ U4m
    
                comm_eta = (U_eta_y + U_m_eta_y) - (U_eta_y + U_m_eta_y).conj().T
    
                # --- Transverse contribution [U_{xy} + U_{-xy} - h.c.] ---
                # U_{xy}(x)
                U1 = Ux[x, y, eta]
                U2 = Uy[xp, y, eta]
                U3 = np.conj(Ux[x, yp, eta].T)
                U4 = np.conj(Uy[x, y, eta].T)
                U_x_y = U1 @ U2 @ U3 @ U4
    
                # U_{-xy}(x)
                U1m = Ux[xm, y, eta].conj().T
                U2m = Uy[xm, y, eta]
                U3m = Ux[xm, yp, eta]
                U4m = np.conj(Uy[x, y, eta].T)
                U_m_x_y = U1m @ U2m @ U3m @ U4m
    
                comm_xy = (U_x_y + U_m_x_y) - (U_x_y + U_m_x_y).conj().T
    
                # --- Aggiorna Ex ---
                Ey_new[x, y, eta] += coeff_eta * comm_eta + coeff_yx * comm_xy

    # --- E^η ---
    coeff_Eeta = (1j * dtau) / (2 * g * a_eta * a_perp**2 * tau)

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                # Shift ciclici
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                # etam = (eta - 1) % Neta

                # Commutatore [Uxη, Uyη]
                # ---- x-direction ----
                U1 = Ux[x, y, eta]
                U2 = Ueta[xp, y, eta]
                U3 = Ux[x, y, etap].conj().T
                U4 = Ueta[x, y, eta].conj().T
                
                U11 = U1 @ U2 @ U3 @ U4
                U11hc = U11.conj().T
                
                U1 = Ux[xm, y, eta].conj().T
                U2 = Ueta[xm, y, eta]
                U3 = Ux[xm, y, etap]
                U4 = Ueta[x, y, eta].conj().T
                
                U22 = U1 @ U2 @ U3 @ U4
                U22hc = U22.conj().T
                
                term_x = U11 + U22 - U11hc - U22hc

                # ---- y-direction ----
                U1 = Uy[x, y, eta]
                U2 = Ueta[x, yp, eta]
                U3 = Uy[x, y, etap].conj().T
                U4 = Ueta[x, y, eta].conj().T
                
                U11 = U1 @ U2 @ U3 @ U4
                U11hc = U11.conj().T
                
                U1 = Uy[x, ym, eta].conj().T
                U2 = Ueta[x, ym, eta]
                U3 = Uy[x, ym, etap]
                U4 = Ueta[x, y, eta].conj().T
                
                U22 = U1 @ U2 @ U3 @ U4
                U22hc = U22.conj().T
                
                term_y = U11 + U22 - U11hc - U22hc

               

                # Totale
                comm = term_x + term_y
                Eeta_new[x, y, eta] += coeff_Eeta * comm

    return Ex_new, Ey_new, Eeta_new


def compute_RHS_YM_at_tau0(Ux, Uy, Ueta, g, a_perp, a_eta, tau0, Nx, Ny, Neta):
    RHS_x = np.zeros((Nx, Ny, Neta, 3, 3), dtype=complex)
    RHS_y = np.zeros((Nx, Ny, Neta, 3, 3), dtype=complex)
    RHS_eta = np.zeros((Nx, Ny, Neta, 3, 3), dtype=complex)

     # --- E^x ---
    coeff_eta = (1j) / (2 * g * a_eta**2 * a_perp * tau0)
    coeff_yx = (1j * tau0) / (2 * g * a_perp**3)
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                etam = (eta - 1) % Neta
    
                # --- Longitudinal contribution [U_{ηx} + U_{-ηx} - h.c.] ---
                # U_{ηx}(x)
                U1 = Ueta[x, y, eta]
                U2 = Ux[x, y, etap]
                U3 = np.conj(Ueta[xp, y, eta].T)
                U4 = np.conj(Ux[x, y, eta].T)
                U_eta_x = U1 @ U2 @ U3 @ U4
    
                # U_{-ηx}(x)
                U1m = Ueta[x, y, etam].conj().T
                U2m = Ux[x, y, etam]
                U3m = Ueta[xp, y, etam]
                U4m = np.conj(Ux[x, y, eta].T)
                U_m_eta_x = U1m @ U2m @ U3m @ U4m
    
                comm_eta = (U_eta_x + U_m_eta_x) - (U_eta_x + U_m_eta_x).conj().T
    
                # --- Transverse contribution [U_{yx} + U_{-yx} - h.c.] ---
                # U_{yx}(x)
                U1 = Uy[x, y, eta]
                U2 = Ux[x, yp, eta]
                U3 = np.conj(Uy[xp, y, eta].T)
                U4 = np.conj(Ux[x, y, eta].T)
                U_y_x = U1 @ U2 @ U3 @ U4
    
                # U_{-yx}(x)
                U1m = Uy[x, ym, eta].conj().T
                U2m = Ux[x, ym, eta]
                U3m = Uy[xp, ym, eta]
                U4m = np.conj(Ux[x, y, eta].T)
                U_m_y_x = U1m @ U2m @ U3m @ U4m
    
                comm_yx = (U_y_x + U_m_y_x) - (U_y_x + U_m_y_x).conj().T

                RHS_x[x, y, eta] += coeff_eta * comm_eta + coeff_yx * comm_yx

    # --- E^y ---
    coeff_eta = (1j) / (2 * g * a_eta**2 * a_perp * tau0)
    coeff_yx = (1j * tau0) / (2 * g * a_perp**3)
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                etam = (eta - 1) % Neta
    
                # --- Longitudinal contribution [U_{ηx} + U_{-ηx} - h.c.] ---
                # U_{ηy}(x)
                U1 = Ueta[x, y, eta]
                U2 = Uy[x, y, etap]
                U3 = np.conj(Ueta[x, yp, eta].T)
                U4 = np.conj(Uy[x, y, eta].T)
                U_eta_y = U1 @ U2 @ U3 @ U4
    
                # U_{-ηy}(x)
                U1m = Ueta[x, y, etam].conj().T
                U2m = Uy[x, y, etam]
                U3m = Ueta[x, yp, etam]
                U4m = np.conj(Uy[x, y, eta].T)
                U_m_eta_y = U1m @ U2m @ U3m @ U4m
    
                comm_eta = (U_eta_y + U_m_eta_y) - (U_eta_y + U_m_eta_y).conj().T
    
                # --- Transverse contribution [U_{xy} + U_{-xy} - h.c.] ---
                # U_{xy}(x)
                U1 = Ux[x, y, eta]
                U2 = Uy[xp, y, eta]
                U3 = np.conj(Ux[x, yp, eta].T)
                U4 = np.conj(Uy[x, y, eta].T)
                U_x_y = U1 @ U2 @ U3 @ U4
    
                # U_{-xy}(x)
                U1m = Ux[xm, y, eta].conj().T
                U2m = Uy[xm, y, eta]
                U3m = Ux[xm, yp, eta]
                U4m = np.conj(Uy[x, y, eta].T)
                U_m_x_y = U1m @ U2m @ U3m @ U4m
    
                comm_xy = (U_x_y + U_m_x_y) - (U_x_y + U_m_x_y).conj().T
    
                RHS_y[x, y, eta] += coeff_eta * comm_eta + coeff_yx * comm_xy

    # --- E^η ---
    coeff_Eeta = (1j) / (2 * g * a_eta * a_perp**2 * tau0)

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                # Shift ciclici
                xp = (x + 1) % Nx
                xm = (x - 1) % Nx
                yp = (y + 1) % Ny
                ym = (y - 1) % Ny
                etap = (eta + 1) % Neta
                # etam = (eta - 1) % Neta

                # Commutatore [Uxη, Uyη]
                # ---- x-direction ----
                U1 = Ux[x, y, eta]
                U2 = Ueta[xp, y, eta]
                U3 = Ux[x, y, etap].conj().T
                U4 = Ueta[x, y, eta].conj().T
                
                U11 = U1 @ U2 @ U3 @ U4
                U11hc = U11.conj().T
                
                U1 = Ux[xm, y, eta].conj().T
                U2 = Ueta[xm, y, eta]
                U3 = Ux[xm, y, etap]
                U4 = Ueta[x, y, eta].conj().T
                
                U22 = U1 @ U2 @ U3 @ U4
                U22hc = U22.conj().T
                
                term_x = U11 + U22 - U11hc - U22hc

                # ---- y-direction ----
                U1 = Uy[x, y, eta]
                U2 = Ueta[x, yp, eta]
                U3 = Uy[x, y, etap].conj().T
                U4 = Ueta[x, y, eta].conj().T
                
                U11 = U1 @ U2 @ U3 @ U4
                U11hc = U11.conj().T
                
                U1 = Uy[x, ym, eta].conj().T
                U2 = Ueta[x, ym, eta]
                U3 = Uy[x, ym, etap]
                U4 = Ueta[x, y, eta].conj().T
                
                U22 = U1 @ U2 @ U3 @ U4
                U22hc = U22.conj().T
                
                term_y = U11 + U22 - U11hc - U22hc

               

                # Totale
                comm = term_x + term_y
                RHS_eta[x, y, eta] += coeff_Eeta * comm
        
    return RHS_x, RHS_y, RHS_eta



def evolve_glasma(Ux, Uy, Ueta, Ex, Ey, Eeta,
                  g, a_perp, a_eta,
                  tau0, dtau, n_steps, nrun,
                  Nx, Ny, N_eta,
                  compute_observable=False,
                  XHQ = None, YHQ = None, etaHQ = None,
                  PXHQ = None, PYHQ = None, PetaHQ=None, 
                  QHQ = None):
    # obs_list = []

    # 0. Pre-update: da τ0 - Δτ a τ0 + Δτ/2
    # tau_init = tau0 - dtau
    # dtau_eff = 1.5 * dtau  # salto effettivo di 3Δτ/2

    tau_cutoff = 1e-24
    include_eta_terms = tau0 > tau_cutoff

    # Ex, Ey, Eeta = update_electric_fields(
    #     Ux, Uy, Ueta, Ex, Ey, Eeta,
    #     g, a_perp, a_eta, tau_init, dtau_eff,
    #     log_step=False,
    #     include_spatial_terms=True,
    #     include_eta_terms=include_eta_terms
    # )

    # da togliere
    # Ex[:] = 0.0
    # Ey[:] = 0.0
# ----------------------------------
    
    
    tau = tau0  # ora inizia il leapfrog  
    


    for step in range(n_steps):
        
        eps_perc = 1e-12
        if step % 25 == 0 or step == n_steps - 1:
            
            perc = 100 * step / (n_steps - 1 + eps_perc)
            print(f"\n[Main][Run {nrun}] Progresso: {step}/{n_steps}  ({perc:.1f}%)")
            sys.stdout.flush()


        
        # 1. Aggiorna E da τ a τ + Δτ/2
        include_eta_terms = tau > tau_cutoff
        
        
        if XHQ is not None:
            assert XHQ.shape == YHQ.shape == PXHQ.shape == PYHQ.shape
            assert QHQ.shape == (XHQ.shape[0], 3, 3)
     

        if i_fluct == 0:
            HQs_active = all(x is not None for x in [XHQ, YHQ, PXHQ, PYHQ, QHQ])
        else:
            HQs_active = all(x is not None for x in [XHQ, YHQ, etaHQ, PXHQ, PYHQ, QHQ])

        if HQs_active and tau>=tau_form:
            
            XHQ, YHQ, etaHQ = evolve_X_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, mass_HQ, tau, dtau, i_fluct)
            
            PXHQ, PYHQ, PetaHQ  = evolve_P_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, QHQ, 
                            Ex, Ey, Eeta, Ux, Uy, Ueta, mass_HQ,
                             tau, dtau, i_fluct)
            
            QHQ = evolve_Q_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, QHQ, 
                            Ex, Ey, Eeta, Ux, Uy, Ueta, mass_HQ,
                             tau, dtau, i_fluct)
            
            # Calcolo della traccia di Q^2 per ogni HQ
            # Q2_trace = np.einsum('nij,njk->nik', QHQ, QHQ)         # Q^2 per ciascun HQ
            # tr_Q2 = np.real(np.trace(Q2_trace, axis1=1, axis2=2))  # Tr(Q^2), shape (NHQ,)
            # mean_tr_Q2 = np.mean(tr_Q2)                            # media su tutti gli HQ
            
            # # Salva su file
            # with open(f"trace_Q2_vs_time_{nrun}.dat", "a") as f:
            #     f.write(f"{tau:.6f}  {mean_tr_Q2:.12e}\n")
                        
           
            
            
#------------------------------------------------------------------------------
#     AGGIUNTA SCRITTURA HQs file finale            
            if step == n_steps-1:
                data = np.column_stack((XHQ, YHQ, hbarc * PXHQ, hbarc * PYHQ))
                fileHQ = f"HQinitializations/XandP_tfin_{nrun}.csv"
                np.savetxt(fileHQ, data, delimiter=",")
                
                print("----------------------------------")
                print(f"FINAL HQs EVENTO {nrun}")
                # print(f"q2: {np.mean(q2):.5f}, deviazione: {np.std(q2):.5e}")
                print("----------------------------------")
            
#------------------------------------------------------------------------------






#   -------- EVOLUZIONE CAMPI ---------


#===========================================================================================================================
# MODE (1) : EVOLUZIONE CON AGGIUNTA FLUTTUAZIONI IN INIZIALIZZAZIONI, se i_fluct non 0
#===========================================================================================================================


#       EVOLUZIONE CON LE PLAQUETTE                
        Ex, Ey, Eeta = update_electric_fields(
            Ux, Uy, Ueta, Ex, Ey, Eeta,
            g, a_perp, a_eta, tau, dtau,
            log_step=False,
            include_spatial_terms=True,
            include_eta_terms=include_eta_terms)
         
        # 2. Aggiorna i gauge links
        Ux, Uy, Ueta = update_gauge_links(Ux, Uy, Ueta, Ex, Ey, Eeta,
                                          g, a_perp, a_eta, tau, dtau)


#===========================================================================================================================
# MODE (2) : EVOLUZIONE CON AGGIUNTA FLUTTUAZIONI IN EVOLUZIONE, se i_fluct non 0
#===========================================================================================================================

#         # #aggiungo le fluttuazioni all'istante tau_fluct_start
#         if step_fluct is not None and step == step_fluct:
#             delta_Ex, delta_Ey, delta_Eeta = delta_fluctuations_glasma(Nx, Ny, N_eta, a_perp, a_eta, Delta, i_fluct, Ux, Uy)
            
#             Ex += delta_Ex
#             Ey += delta_Ey
#             Eeta += delta_Eeta

#             # traccia_eeta_sf = calculate_trace_field(Eeta) * hbarc**4
#             # traccia_f = calculate_trace_field(delta_Eeta) * hbarc**4
#             # traccia_epf = calculate_trace_field(Eeta) * hbarc**4
#             # print(f"traccia fluttuazione: {traccia_f}")
#             # print(f"traccia campo sommato: {traccia_epf}")
#             # print(f"traccia campo senza flutt: {traccia_eeta_sf}")

# #       EVOLUZIONE CON LE PLAQUETTE                
#         Ex, Ey, Eeta = update_electric_fields(
#             Ux, Uy, Ueta, Ex, Ey, Eeta,
#             g, a_perp, a_eta, tau, dtau,
#             log_step=False,
#             include_spatial_terms=True,
#             include_eta_terms=include_eta_terms)
         
#         # 2. Aggiorna i gauge links
#         Ux, Uy, Ueta = update_gauge_links(Ux, Uy, Ueta, Ex, Ey, Eeta,
#                                           g, a_perp, a_eta, tau, dtau)


        if step == step_target and nrun == 0:
            epsilon_wri, PT_wri, PL_wri = calculate_spatial_observables(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau)
            export_spatial_observables_eta0(x, y, epsilon_wri, PT_wri, PL_wri, step)
            
        if step == step_target:
            epsilon_wri, PT_wri, PL_wri = calculate_spatial_observables(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau)
            
            #epsilon_2D = np.squeeze(epsilon_wri)  # diventa (Nx, Ny); questo comando va bene solo se N_eta=1
            epsilon_2D = np.mean(epsilon_wri, axis=2)
            
        
            eps2 = compute_eccentricity_2(X, Y, epsilon_2D)
            
            print(f"[nrun = {nrun:.4f}]     [epsilon_2 = {eps2:.4f}]")
            with open(f"evolution_run/evolution_run{nrun}/eccentricity.csv", "w") as f:
                f.write(f"{eps2}\n")

        # 3. Osservabili
        if compute_observable and (step == 0 or step % step_interval == 0):
        #if compute_observable and step > 0 and (step % step_interval == 0):

            epsilon_wri, PT_wri, PL_wri = calculate_spatial_observables(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau)

            # if step==0:
            #     tr_ex = calculate_trace_field(Ex,epsilon_wri)
            #     tr_ey = calculate_trace_field(Ey,epsilon_wri)
            #     print(f"Traccia dei campi trasversi post prima evoluzione (eps weight)={tr_ex} e {tr_ey}")

            tr_ex = calculate_trace_field(Ex,epsilon_wri)
            tr_ey = calculate_trace_field(Ey,epsilon_wri)
            tr_eta = calculate_trace_field(Eeta,epsilon_wri)
            # tr_ex = calculate_trace_field(Ex)
            # tr_ey = calculate_trace_field(Ey)
            # tr_eta = calculate_trace_field(Eeta)
            energy_electric = (tr_ex + tr_ey) / tau**2 + tr_eta
            
             

            obs_ex = tr_ex * hbarc**4
            obs_ey = tr_ey * hbarc**4
            obs_eta = tr_eta * hbarc**4
            obs_energy_electric = energy_electric * hbarc**4
            
            
            # Campi magnetici
            BL2, BT2, energy_magnetic = calculate_magnetic_energy(Ux, Uy, Ueta, g, a_perp, a_eta, tau,epsilon_wri)
            # weight_test = np.ones((Nx, Ny, N_eta))    #serve per fare la media geometrica sul reticolo
            # BL2, BT2, energy_magnetic = calculate_magnetic_energy(Ux, Uy, Ueta, g, a_perp, a_eta, tau,weight_test)
            obs_BL2 = BL2 * hbarc**4
            obs_BT2 = BT2 * hbarc**4
            obs_energy_magnetic = energy_magnetic * hbarc**4
            
            

            obs_energy_tot=obs_energy_electric + obs_energy_magnetic
            obs_PT =  obs_eta + obs_BL2
            obs_PL = obs_energy_tot -2*obs_PT
            
            # # DEBUG - solo al primo step
            #if step == 0:
            #     print(f"obs_PT = {obs_PT:.4e}")
            #     print(f"obs_PL = {obs_PL:.4e}")
            #     print(f"PL/PT = {obs_PL/obs_PT:.4e}")
            #     print(f"obs_eta = {obs_eta:.4e}")
            #     print(f"obs_BL2 = {obs_BL2:.4e}")
            #    print(f"obs_BT2 = {obs_BT2:.4e}")
            #     print(f"obs_energy_electric = {obs_energy_electric:.4e}")
            #     print(f"obs_energy_magnetic = {obs_energy_magnetic:.4e}")
            #     print(f"obs_energy_tot = {obs_energy_tot:.4e}")
            # if step==1:
            #     print(f"obs_BT2_secondo_step = {obs_BT2:.4e}")


            #epsilon_2D = np.squeeze(epsilon_wri)  # diventa (Nx, Ny); questo comando va bene solo se N_eta=1
            epsilon_2D = np.mean(epsilon_wri, axis=2)
            eps2 = compute_eccentricity_2(X, Y, epsilon_2D)
            eps3 = compute_eccentricity_3(X, Y, epsilon_2D)
            eps2T = compute_eccentricity_eps2T_w(Ux, Uy, Ueta, Ex, Ey, Eeta, 
                                                 g, a_perp, a_eta, tau, 
                                                 epsilon_wri)
            
            print(f"[nrun = {nrun:.4f}]     [τ = {tau:.4f}]    epsil = {obs_energy_tot:.4e}")
            print(f"[nrun = {nrun:.4f}]     [epsilon_2 = {eps2:.4f}]")
            print(f"[nrun = {nrun:.4f}]     [epsilon_3 = {eps3:.4f}]")
            
            # Scrittura su CSV
            csv_writer.writerow([
                tau-tau0,
                obs_ex / tau**2, obs_ey/ tau**2, obs_eta,
                obs_energy_electric,
                obs_BL2, obs_BT2, obs_energy_tot,obs_PT, obs_PL, eps2, eps3,
                eps2T
                ])
            # obs_list.append(obs_eta,obs_energy_electric)



        if step == n_steps-1: 

            tolle = 1e-1
#------------------------------------------------------------------------------
# iter_OR = 0: NO proiezione trasversa discreta
# iter_OR = 1: proiezione trasversa discreta
#------------------------------------------------------------------------------

                  
            
            alphaRELAX = 0.7    # inutile al momento
            compute_spettro(alphaRELAX, 
                                Ux, Uy, Ueta, Ex, Ey, Eeta, nrun, 
                                tau, a_perp, g, 
                                tolle, iter_OR)

            print(f"[step {step}] Calcolo spettro finale con tol = {tolle:.1e}")


        tau += dtau

    # if compute_observable:
    #     return Ux, Uy, Ueta, Ex, Ey, Eeta, obs_list
    # else:
    #     return Ux, Uy, Ueta, Ex, Ey, Eeta
    
    
    # return Ux, Uy, Ueta, Ex, Ey, Eeta


    if XHQ is not None:
        return Ux, Uy, Ueta, Ex, Ey, Eeta, XHQ, YHQ, PXHQ, PYHQ, PetaHQ, QHQ
    else:
        return Ux, Uy, Ueta, Ex, Ey, Eeta




#------------------------------------------------------------------------------
# funzioni ACCESSORIE
#------------------------------------------------------------------------------



def calculate_plaquette_perB(Ux, Uy, Ueta):
    Nx, Ny, Neta = Ux.shape[:3]

    Uxy = np.zeros_like(Ux, dtype=np.complex128)
    Uetax = np.zeros_like(Ux, dtype=np.complex128)
    Uetay = np.zeros_like(Ux, dtype=np.complex128)

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):

                # Uxy plaquette = Ux(x) Uy(x+hat_x) Ux†(x+hat_y) Uy†(x)
                U1 = Ux[x, y, eta]
                U2 = Uy[(x+1)%Nx, y, eta]
                U3 = Ux[x, (y+1)%Ny, eta].conj().T
                U4 = Uy[x, y, eta].conj().T
                Uxy[x, y, eta] = U1 @ U2 @ U3 @ U4

                # Uetax plaquette 
                U1 = Ueta[x, y, eta]
                U2 = Ux[x, y, (eta+1)%Neta]
                U3 = Ueta[(x+1)%Nx, y, eta].conj().T
                U4 = Ux[x, y, eta].conj().T
                Uetax[x, y, eta] = U1 @ U2 @ U3 @ U4

                # Uetay plaquette 
                U1 = Ueta[x, y, eta]
                U2 = Uy[x, y, (eta+1)%Neta]
                U3 = Ueta[x, (y+1)%Ny, eta].conj().T
                U4 = Uy[x, y, eta].conj().T
                Uetay[x, y, eta] = U1 @ U2 @ U3 @ U4
                
                

    return Uxy, Uetax, Uetay





def calculate_spatial_observables(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, tau):
    """
    Calcola l'energia totale, P_T e P_L in ogni punto del reticolo (x, y, eta),
    restituendo array 3D.

    Parametri:
    - Ux, Uy, Ueta: link gauge 3x3
    - Ex, Ey, Eeta: campi elettrici 3x3
    - g: costante di accoppiamento
    - a_perp, a_eta: passi reticolari
    - tau: tempo proprio

    Ritorna:
    - epsilon (Nx, Ny, Neta)
    - PT      (Nx, Ny, Neta)
    - PL      (Nx, Ny, Neta)
    """

    assert Ex.shape[-2:] == (3, 3), "Campi E devono essere matrici 3x3"
    Nx, Ny, Neta = Ex.shape[:3]

    # Plaquette magnetiche
    Uxy, Uetax, Uetay = calculate_plaquette_perB(Ux, Uy, Ueta)

    BL2 = np.zeros((Nx, Ny, Neta))
    BT2 = np.zeros((Nx, Ny, Neta))

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                # Magnetici
                BL2[x, y, eta] = np.trace(np.eye(3) - Uxy[x, y, eta]).real
                BT2[x, y, eta] = (
                    np.trace(np.eye(3) - Uetax[x, y, eta]).real +
                    np.trace(np.eye(3) - Uetay[x, y, eta]).real
                )

    # Normalizzazione magnetica corretta
    BL2 *= 2.0 / (g**2 * a_perp**4)
    BT2 *= 2.0 / ((g * a_eta * a_perp * tau)**2)

    # Campi elettrici
    EL2 = np.zeros((Nx, Ny, Neta))
    ET2 = np.zeros((Nx, Ny, Neta))

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                EL2[x, y, eta] = np.trace(Eeta[x, y, eta] @ Eeta[x, y, eta]).real
                ET2[x, y, eta] = (
                    np.trace(Ex[x, y, eta] @ Ex[x, y, eta]).real +
                    np.trace(Ey[x, y, eta] @ Ey[x, y, eta]).real
                )

    ET2 *= 1.0 / tau**2
    # EL2 *= 1 → non serve moltiplicare, è già corretto

    # Osservabili locali
    epsilon = EL2 + ET2 + BL2 + BT2
    PT      = EL2 + BL2
    PL      = -EL2 + ET2 - BL2 + BT2

    return epsilon, PT, PL


def epsilon_weighted(quantity, epsilon):
    """
    Calcola la media pesata della quantity usando epsilon come peso.
    quantity ed epsilon devono avere shape compatibili, es: (Nx, Ny, Neta)
    """
    weight = epsilon
    weight_sum = np.sum(weight)
    if weight_sum == 0:
        return 0.0  # oppure np.nan, a seconda di cosa preferisci in casi patologici
    return np.sum(quantity * weight) / weight_sum



def calculate_trace_field(E, WEIGHT=None, average=True):    # traccia su colore per shapes Nx Ny N_eta

    """
    Calcola Tr(E E) in ogni punto del reticolo 3D, stile esplicito (dot + trace).

    Parametri:
    - E: array di shape (Nx, Ny, Neta, 3, 3)
    - average: se True, restituisce la media sul volume; altrimenti array 3D

    Ritorna:
    - valore medio o array con tracce punto per punto (parte reale)
    """

    Nx, Ny, Neta = E.shape[:3]
    trace = np.zeros((Nx, Ny, Neta), dtype=np.complex128)
    
    
    
    # global Tp_weighted
    # Normalizzazione 
    normalization = np.sum(WEIGHT) if WEIGHT is not None else Nx * Ny * Neta



    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                E_squared = np.dot(E[x, y, eta], E[x, y, eta])
                # trace[x, y, eta] = np.trace(E_squared)
                w = WEIGHT[x, y, eta] if WEIGHT is not None else 1.0
                trace[x, y, eta] = w * np.trace(E_squared) / normalization


    trace_real = trace.real  # Traccia fisica

    if average:
        # return np.mean(trace_real)
        return np.sum(trace_real)          # già normalizzata con Tp
    else:
        return trace_real



def calculate_magnetic_energy(Ux, Uy, Ueta, g, a_perp, a_eta, tau, WEIGHT):
    # Prende le tracce delle plaquette già implementate
    tr_Uxy, tr_Uetax, tr_Uetay = calculate_plaquette_perB(Ux, Uy, Ueta)


    Nx, Ny, Neta = Ux.shape[:3]   

    # global Tp_weighted
    # Normalizzazione 
    norm = np.sum(WEIGHT)
     


    

    BL2 = 0.0
    BT2 = 0.0

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                

                # Traccia (𝟙 - U_xy)
                tr_bl = np.trace(np.eye(3) - tr_Uxy[x, y, eta]).real
                BL2 += WEIGHT[x,y,eta] * tr_bl

                # Traccia (𝟙 - U_eta_i), i = x, y
                tr_bt_x = np.trace(np.eye(3) - tr_Uetax[x, y, eta]).real
                tr_bt_y = np.trace(np.eye(3) - tr_Uetay[x, y, eta]).real
                BT2 += WEIGHT[x,y,eta] * (tr_bt_x + tr_bt_y)

    BL2 *= (2.0 / (g**2 * a_perp**4)) / norm
    BT2 *= (2.0 / ((g * a_eta * a_perp * tau)**2)) / norm

    energy_magnetic = BL2 + BT2
    return BL2, BT2, energy_magnetic




def export_spatial_observables_eta0(x, y, epsilon, PT, PL, step):
    """
    Esporta (x, y, eta, ε_tot, P_T, P_L) sul piano η = 0, senza header nel file CSV.
    """
    NNeta = epsilon.shape[2]
    k = NNeta // 2  # η = 0

    filename = f"observables_eta0_step{step}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        Nx, Ny = epsilon.shape[:2]
        for i in range(Nx):
            for j in range(Ny):
                writer.writerow([
                    x[i],
                    y[j],
                    epsilon[i, j, k] * hbarc**4,
                    PT[i, j, k] * hbarc**4,
                    PL[i, j, k] * hbarc**4
                ])

    print(f"[CSV] Salvato piano η=0 → {filename} (senza header)")
    
    
def export_single_Tp(x, y, Tp, iproie):
    """
    Esporta una matrice Tp (x, y, Tp) su file CSV (senza header).
    """
    Nx, Ny = Tp.shape
    nome_file = f"tp{iproie}.csv"
    with open(nome_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(Nx):
            for j in range(Ny):
                writer.writerow([
                    x[i],
                    y[j],
                    Tp[i, j]
                ])
    print(f"[CSV] Tp esportata in {nome_file}")
    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def plot_quantity(X, Y, field, title, cmap="plasma", label="", hotspots=None, levels=50):
    """
    Plot 2D di un campo scalare su griglia.

    Parametri:
    - X, Y: griglie coordinate 2D
    - field: array 2D (quantità da plottare)
    - title: titolo del plot
    - cmap: colormap da usare (default: "viridis")
    - label: etichetta per colorbar (es. "$T_p(x, y)$")
    - hotspots: lista di (x, y) da plottare come pallini rossi (default: None)
    - levels: numero di livelli nel contourf
    """

    plt.figure(figsize=(10, 8))
    cont = plt.contourf(X, Y, field, levels=levels, cmap=cmap)
    plt.colorbar(cont, label=label)
    plt.xlabel("x [fm]")
    plt.ylabel("y [fm]")
    plt.title(title)

    if hotspots is not None:
        x_h, y_h = zip(*hotspots)
        plt.scatter(x_h, y_h, color='red', s=60, edgecolor='black', zorder=3)

    plt.tight_layout()
    # plt.savefig("tp.png", dpi=300)
    # plt.savefig("tp.pdf")
    plt.show()
    
    
def plot_rho_component(rho, a, X, Y, cmap="coolwarm", levels=50):
    """
    Plot contour della componente rho^a(x, y) per un dato a ∈ [1,8].

    Parametri:
    - rho: array con shape (9, Nx, Ny) (rho[0] = 0, rho[1:]=fisiche)
    - a: componente da plottare (1-8)
    - X, Y: griglie 2D delle coordinate fisiche
    - cmap: colormap per il plot
    - levels: numero di livelli nel contour
    """
    if a < 1 or a > 8:
        raise ValueError("Indice colore a deve essere compreso tra 1 e 8")

    plt.figure(figsize=(7, 6))
    cont = plt.contourf(X, Y, rho[a], levels=levels, cmap=cmap)
    plt.colorbar(cont, label=fr"$\rho^{a}(x, y)$ [fm$^{{-2}}$]")
    plt.xlabel("x [fm]")
    plt.ylabel("y [fm]")
    plt.title(fr"Contour plot di $\rho^{a}(x, y)$")
    plt.tight_layout()
    plt.show()
    
def plot_alpha_component(alpha, a, X, Y, cmap="coolwarm", levels=50):
    """
    Plot contour della componente alpha^a(x, y) per un dato a ∈ [1,8].

    Parametri:
    - rho: array con shape (9, Nx, Ny) (rho[0] = 0, rho[1:]=fisiche)
    - a: componente da plottare (1-8)
    - X, Y: griglie 2D delle coordinate fisiche
    - cmap: colormap per il plot
    - levels: numero di livelli nel contour
    """
    if a < 1 or a > 8:
        raise ValueError("Indice colore a deve essere compreso tra 1 e 8")

    plt.figure(figsize=(7, 6))
    cont = plt.contourf(X, Y, alpha[a], levels=levels, cmap=cmap)
    plt.colorbar(cont, label=fr"$\alpha^{a}(x, y)$ ")
    plt.xlabel("x [fm]")
    plt.ylabel("y [fm]")
    plt.title(fr"Contour plot di $\alpha^{a}(x, y)$")
    plt.tight_layout()
    plt.show()    

def check_wilson_unitarity(V):
    """
    Controlla la proprietà unitaria delle matrici di Wilson V(x, y).
    Verifica che V * V† = I e che il determinante sia 1.

    Parametri:
    - V: array shape (Nx, Ny, 3, 3) contenente le matrici di Wilson per ogni punto (x, y).

    Restituisce:
    - None: stampa il risultato dei controlli.
    """
    Nx, Ny = V.shape[:2]

    # Punti di controllo scelti (posizioni specifiche sulla griglia)
    points_to_check = [
        (0, 0),  # angolo in alto a sinistra
        (Nx//2, Ny//2),  # centro
        (Nx-1, Ny-1)  # angolo in basso a destra
    ]

    # Aggiungi punti casuali (ad esempio 5 punti casuali sulla griglia)
    np.random.seed(42)  # Per avere riproducibilità
    random_points = [(np.random.randint(0, Nx), np.random.randint(0, Ny)) for _ in range(5)]
    points_to_check.extend(random_points)

    for i, j in points_to_check:
        V_dagger = np.conj(V[i, j].T)  # Calcola V† (dagger di V)
        identity_check = np.allclose(np.dot(V[i, j], V_dagger), np.eye(3))  # Verifica che V * V† = I
        det_check = np.isclose(np.linalg.det(V[i, j]), 1)  # Verifica che det(V) = 1

        print(f"Controllo punto ({i}, {j}):")
        print(f"  V * V† = Identity? {'Sì' if identity_check else 'No'}")
        print(f"  det(V) = 1? {'Sì' if det_check else 'No'}")

    print("Controlli completati!")


def check_wilson_unitarity_full(V):
    """
    Controlla la proprietà unitaria di tutte le matrici di Wilson V(x, y).
    Verifica che V * V† = I per ogni punto della griglia.

    Parametri:
    - V: array shape (Nx, Ny, 3, 3) contenente le matrici di Wilson per ogni punto (x, y).

    Restituisce:
    - None: stampa il risultato dei controlli.
    """
    Nx, Ny = V.shape[:2]
    identity_check_all = np.zeros((Nx, Ny), dtype=bool)  # Array per tracciare la correttezza

    for i in range(Nx):
        for j in range(Ny):
            V_dagger = np.conj(V[i, j].T)  # Calcola V† (dagger di V)
            identity_check_all[i, j] = np.allclose(np.dot(V[i, j], V_dagger), np.eye(3))  # Verifica che V * V† = I

    # Plot 3D per visualizzare i controlli
    
   

    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, identity_check_all.astype(int), cmap="viridis")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('V * V† = Identity?')
    ax.set_title('Controllo della proprietà unitaria su tutta la griglia')

    plt.show()

    print("Controllo completato per tutte le matrici di Wilson.")





def test_su3_normalization(t_a):
    """
    Testa la normalizzazione delle tracce dei generatori SU(3) t^a t^b.
    Si aspetta che Tr(t^a t^b) = (1/2) delta_ab.

    Parametri:
    - t_a: array (8, 3, 3) che contiene i generatori SU(3).
    """
    Nc = t_a.shape[0]  # Numero di generatori (8 per SU(3))
    for a in range(Nc):
        for b in range(Nc):
            trace_value = np.trace(np.dot(t_a[a], t_a[b]))  # Tr(t^a t^b)
            trace_value = np.real(trace_value)  # Prendiamo solo la parte reale
            expected_value = 0.5 if a == b else 0.0  # Deve essere 1/2 se a == b, altrimenti 0
            print(f"Tr(t^{a+1} t^{b+1}) = {trace_value:.5f}, Expected = {expected_value:.5f}")
            assert np.isclose(trace_value, expected_value), f"Errore nella normalizzazione di Tr(t^{a+1} t^{b+1})"
    
    print("Test di normalizzazione delle tracce passato con successo!")
    
    

def print_Eeta_trace_stats(Eeta_trace):
    """
    Stampa statistiche (mean, max, min, std) per la parte reale e immaginaria
    della matrice Eeta_trace (Nx x Ny).
    """
    Eeta_real = np.real(Eeta_trace)
    # Eeta_imag = np.imag(Eeta_trace)

    print("Statistiche per Eeta_trace (parte REALE):")
    print(f"  Mean = {np.mean(Eeta_real) * hbarc**4} GeV$^4$")
    print(f"  Max  = {np.max(Eeta_real) * hbarc**4} GeV$^4$")
    print(f"  Min  = {np.min(Eeta_real) * hbarc**4} GeV$^4$")
    
    init_EeleEele=np.mean(Eeta_real) * hbarc**4
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"testglasma/glasma_initener_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(f"Initial electric energy density: {init_EeleEele:.6e} GeV⁴\n")


    # print("\nStatistiche per Eeta_trace (parte IMMAGINARIA):")
    # print(f"  Mean = {np.mean(Eeta_imag)* hbarc**4}")
    # print(f"  Max  = {np.max(Eeta_imag)* hbarc**4}")
    # print(f"  Min  = {np.min(Eeta_imag)* hbarc**4}")
     
    print("-"*50)


def get_time_seed():
    """
    Restituisce un intero derivato dall'orario corrente (ora, minuti, secondi, microsecondi),
    da usare come seed per il random.
    """
    now = datetime.datetime.now()
    # Concatena hour, minute, second, microsecond in un'unica stringa, poi converte in int
    # Esempio formato: HHMMSSffffff (6 cifre microsecondi)
    seed_str = now.strftime("%H%M%S%f")  
    # Se vuoi ridurlo per sicurezza entro 2^32 (range seed standard)
    seed_int = int(seed_str) % (2**32)
    return seed_int





#------------------------------------------------------------------------------
# FUNZIONI PER CHECK
#------------------------------------------------------------------------------    

def compute_covariant_divergence(U, E, direction, a):
    """
    Calcola il divergente covariante D^μ E^μ lungo una direzione (x, y o eta).

    Parametri:
    - U: gauge link lungo μ (Nx, Ny, Neta, 3, 3)
    - E: campo elettrico lungo μ (Nx, Ny, Neta, 3, 3)
    - direction: 'x', 'y' o 'eta'
    - a: passo reticolare (a_perp o a_eta)

    Ritorna:
    - D^μ E^μ come array (Nx, Ny, Neta, 3, 3)
    """
    shift = {'x': (1, 0, 0), 'y': (0, 1, 0), 'eta': (0, 0, 1)}[direction]
    U_dag = np.transpose(np.conj(U), axes=(0, 1, 2, 4, 3))

    # Shift E indietro
    E_back = np.roll(E, shift=[-s for s in shift], axis=(0, 1, 2))
    
    

    # D^μ E^μ = (U(x) E(x+μ) U†(x) - E(x)) / a
    transported = np.einsum('...ij,...jk,...kl->...il', U, E_back, U_dag)
    divergence = (transported - E) / a

    return divergence

def check_gauss_law(Ux, Uy, Ueta, Ex, Ey, Eeta, a_perp, a_eta):
    DxEx = compute_covariant_divergence(Ux, Ex, 'x', a_perp)
    DyEy = compute_covariant_divergence(Uy, Ey, 'y', a_perp)
    DetaEeta = compute_covariant_divergence(Ueta, Eeta, 'eta', a_eta)

    total = DxEx + DyEy + DetaEeta  # Shape (Nx, Ny, Neta, 3, 3)

    # Calcola norma gauge-invariante del tensore (Tr[(D^μ E^μ)^2])
    residual = np.real(np.einsum('...ij,...ji->...', total, total))

    mean = np.mean(np.abs(residual))
    maxval = np.max(residual)

    return residual, mean, maxval

def check_gauss(Ux, Uy, Ueta, Ex, Ey, Eeta, a_perp, a_eta):
    DxEx = compute_covariant_divergence(Ux, Ex, 'x', a_perp)
    DyEy = compute_covariant_divergence(Uy, Ey, 'y', a_perp)
    DetaEeta = compute_covariant_divergence(Ueta, Eeta, 'eta', a_eta)

    total = DxEx + DyEy + DetaEeta  # Shape (Nx, Ny, Neta, 3, 3)

    # Calcola norma gauge-invariante del tensore (Tr[(D^μ E^μ)^2])
    residual = np.real(np.einsum('...ij,...ji->...', total, total))

    

    return residual


def cool_electric_fields(Ux, Uy, Ueta, Ex, Ey, Eeta, a_perp, a_eta, cooling_rate=0.1):
    """
    Raffredda i campi elettrici per ridurre la violazione della legge di Gauss.
    
    Parametri:
    - Ux, Uy, Ueta: gauge links
    - Ex, Ey, Eeta: campi elettrici
    - a_perp, a_eta: spacing reticolare
    - cooling_rate: frazione del gradiente da sottrarre (default: 0.1)
    
    Ritorna:
    - Ex_new, Ey_new, Eeta_new: campi corretti
    """
    residual = check_gauss(Ux, Uy, Ueta, Ex, Ey, Eeta, a_perp, a_eta)


    cooling_add = 0.001  # o anche più piccolo, tipo 0.001 per essere conservativi
    dphi_dx = cooling_add * compute_covariant_derivative(Ux, residual, 'x', a_perp)
    dphi_dy = cooling_add * compute_covariant_derivative(Uy, residual, 'y', a_perp)
    dphi_deta = cooling_add * compute_covariant_derivative(Ueta, residual, 'eta', a_eta)

    Ex_new = Ex - cooling_rate * dphi_dx
    Ey_new = Ey - cooling_rate * dphi_dy
    Eeta_new = Eeta - cooling_rate * dphi_deta
    
    
    return Ex_new, Ey_new, Eeta_new

def compute_covariant_derivative(U, phi, direction, a):
    """
    Calcola la derivata covariante D^μ φ lungo una direzione (x, y o eta).
    
    Parametri:
    - U: gauge link lungo μ (Nx, Ny, Neta, 3, 3)
    - phi: campo scalare (Nx, Ny, Neta)
    - direction: 'x', 'y' o 'eta'
    - a: passo reticolare
    
    Ritorna:
    - derivata covariante D^μ φ come array (Nx, Ny, Neta, 3, 3)
    """
    shift = {'x': (1, 0, 0), 'y': (0, 1, 0), 'eta': (0, 0, 1)}[direction]

    # Shift scalare phi in avanti
    phi_forward = np.roll(phi, shift=shift, axis=(0, 1, 2))

    # Trasporto parallelo del gradiente: U(x) * [φ(x+μ) - φ(x)]
    delta_phi = phi_forward - phi

    # Moltiplicazione con U(x): estende il campo scalare a una matrice
    delta_phi_mat = delta_phi[:, :, :, np.newaxis, np.newaxis] * np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :]

    cov_derivative = np.einsum('...ij,...jk->...ik', U, delta_phi_mat) / a

    return cov_derivative


def assert_parseval(rho_x, rho_k, tol=1e-10):
    """
    Verifica il teorema di Parseval:
        sum |rho(x)|^2 ≈ sum |rho(k)|^2

    Parametri:
    - rho_x: array nello spazio delle coordinate
    - rho_k: array in k-space (FFT di rho_x)
    - tol: tolleranza relativa ammessa

    Se la differenza è significativa, solleva AssertionError.
    """
    energy_x = np.sum(np.abs(rho_x)**2)
    energy_k = np.sum(np.abs(rho_k)**2)

    rel_diff = abs(energy_x - energy_k) / energy_x
    if rel_diff > tol:
        raise AssertionError(
            f"[Parseval ERROR] Energia non conservata: "
            f"x-space = {energy_x:.5e}, k-space = {energy_k:.5e}, "
            f"diff = {rel_diff:.2e}"
        )


#------------------------------------------------------------------------------
#  eccentricità eps2
#------------------------------------------------------------------------------ 

def compute_eccentricity_2(X, Y, epsilon_2D):
# la epsilon_2D deve avere shape Nx x Ny
    """Calcola eccentricità ε₂ nel piano trasverso"""
    weight = epsilon_2D / np.sum(epsilon_2D)

    x2_mean = np.sum((X**2) * weight)
    y2_mean = np.sum((Y**2) * weight)
    xy_mean = np.sum((X * Y) * weight)

    numerator = np.sqrt((x2_mean - y2_mean)**2 + (2 * xy_mean)**2)
    denominator = x2_mean + y2_mean

    eps2 = numerator / denominator
    return eps2


def compute_eccentricity_3(X, Y, epsilon_2D):
# la epsilon_2D deve avere shape Nx x Ny
    """Calcola eccentricità ε₂ nel piano trasverso"""
    weight = epsilon_2D / np.sum(epsilon_2D)

    x3_mean = np.sum((X**3) * weight)
    y3_mean = np.sum((Y**3) * weight)
    xy2_mean = np.sum((X  *   Y**2) * weight)
    x2y_mean = np.sum((X**2  *   Y) * weight)
    rT = np.sqrt(X**2  +   Y**2)
    r3 = rT**3
    r3_mean = np.sum(r3  * weight)
    
    numeratorF = (x3_mean - 3 * xy2_mean)**2 + (y3_mean - 3 * x2y_mean)**2
    numerator = np.sqrt( numeratorF )
    denominator = r3_mean

    eps3 = numerator / denominator
    return eps3



def compute_eccentricity_eps2T_w(Ux, Uy, Ueta, Ex, Ey, Eeta, g, a_perp, a_eta, 
                                 tau, WEIGHT):
    """
    Calcola la media sul reticolo di eps2T.

    Parametri:
    - Ux, Uy, Ueta: link gauge 3x3
    - Ex, Ey, Eeta: campi elettrici 3x3
    - g: costante di accoppiamento
    - a_perp, a_eta: passi reticolari
    - tau: tempo proprio

    Ritorna:
    - epsilon2T
    """

    assert Ex.shape[-2:] == (3, 3), "Campi E devono essere matrici 3x3"
    Nx, Ny, Neta = Ex.shape[:3]

    # Plaquette magnetiche
    Uxy, Uetax, Uetay = calculate_plaquette_perB(Ux, Uy, Ueta)
    
     
    # normalization = np.mean(WEIGHT)

# CALCOLO di Tr(Bx^2) e Tr(By^2)

    TrBx2 = np.zeros((Nx, Ny, Neta))
    TrBy2 = np.zeros((Nx, Ny, Neta))

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                # Magnetici
                TrBx2[x, y, eta] = (
                    np.trace(np.eye(3) - Uetay[x, y, eta]).real
                )
                TrBy2[x, y, eta] = (
                    np.trace(np.eye(3) - Uetax[x, y, eta]).real 
                )

    # Normalizzazione magnetica corretta
    TrBx2 *= 2.0  / ((g * a_eta * a_perp * tau)**2) # Tr(Bx^2) 
    TrBy2 *= 2.0 / ((g * a_eta * a_perp * tau)**2) # Tr(By^2) 


# CALCOLO di Bx e By

    TBa_x = np.zeros((Nx, Ny, Neta, 3, 3), dtype=complex)
    TBa_y = np.zeros((Nx, Ny, Neta, 3, 3), dtype=complex)
    
    t_a = get_su3_generators()  # shape: (8, 3, 3)
    pref = 2.0 / (1j * g * a_eta * a_perp * tau)
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                delta_Uetay = np.eye(3) - Uetay[x, y, eta]
                delta_Uetax = np.eye(3) - Uetax[x, y, eta]
                
                # Somma su a: T^a * Tr[T^a (1 - U)]
                for a in range(8):
                    tr_x = np.trace(t_a[a] @ delta_Uetay)
                    tr_y = np.trace(t_a[a] @ delta_Uetax)
                    
                    TBa_x[x, y, eta] += t_a[a] * tr_x
                    TBa_y[x, y, eta] += t_a[a] * tr_y
    
    # Normalizzazione finale
    TBa_x *= pref   # Bx 
    TBa_y *= -pref   # By


# Tr(BxBy)/tau^2

    TrBxBy = np.zeros((Nx, Ny, Neta))
    
    
    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                
                TrBxBy[x, y, eta] = (
                    np.trace(TBa_x[x, y, eta] @ TBa_y[x, y, eta]).real 
                )
               
    # TrBxBy *= 1 


# Tr(Ex^2)/tau^2 e Tr(Ey^2)/tau^2
    TrEx2 = np.zeros((Nx, Ny, Neta))
    TrEy2 = np.zeros((Nx, Ny, Neta))

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                
                TrEx2[x, y, eta] = (
                    np.trace(Ex[x, y, eta] @ Ex[x, y, eta]).real 
                ) / tau**2
                TrEy2[x, y, eta] = (
                    np.trace(Ey[x, y, eta] @ Ey[x, y, eta]).real
                ) / tau**2

    # TrEx2 *= WEIGHT
    # TrEy2 *= WEIGHT

# Tr(ExEy)/tau^2
    
    TrExEy = np.zeros((Nx, Ny, Neta))

    for x in range(Nx):
        for y in range(Ny):
            for eta in range(Neta):
                
                TrExEy[x, y, eta] = (
                    np.trace(Ex[x, y, eta] @ Ey[x, y, eta]).real 
                ) / tau**2
                

    # TrExEy *= WEIGHT

    Txx = WEIGHT - 2 * (TrEx2 + TrBx2)
    Tyy = WEIGHT - 2 * (TrEy2 + TrBy2)
    Txy = - 2. * (TrExEy + TrBxBy)
    
    mean_Txx = np.mean(Txx * WEIGHT)
    mean_Tyy = np.mean(Tyy * WEIGHT)
    mean_Txy = np.mean(Txy * WEIGHT)
    numeratore = (mean_Txx - mean_Tyy)**2 + 4 * mean_Txy**2
    epsreg = 1e-12
    denominatore = mean_Txx + mean_Tyy + epsreg
    
    # denominatore = np.mean(WEIGHT)
    
    epsilon2T = np.sqrt(numeratore) / denominatore
    
    
    

    return epsilon2T



def compute_spettro(alOR, Ux, Uy, Ueta, Ex, Ey, Eeta, event_id, 
                    proptime, aperp, gcoupl, 
                    tol, niter_OR):

#------------------------------------------------------------------------------
# La routine calcola lo spettro nel Coulomb gauge.
# Nell'attuale implementazione, se niter_OR=0 allora non c'è alcuna proiezione
# sui campi trasversi. Se invece niter_OR=1, la routine calcola la
# proiezione col proiettore trasverso definito sul reticolo.
#------------------------------------------------------------------------------    
    
#------------------------------------------------------------------------------
# SLICE in eta
#------------------------------------------------------------------------------
    neta_slice = neta_zero
    
    Ux2 = Ux[:, :, neta_slice, :, :]
    Uy2 = Uy[:, :, neta_slice, :, :]
    Ex2 = Ex[:, :, neta_slice, :, :]
    Ey2 = Ey[:, :, neta_slice, :, :]
    Ueta2 = Ueta[:, :, neta_slice, :, :]
    Eeta2 = Eeta[:, :, neta_slice, :, :]

#------------------------------------------------------------------------------
# ESTRAZIONE Ai e Aeta dai gauge links
#------------------------------------------------------------------------------
    
    Ax = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    Ay = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    Aeta = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    # theta = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    
    
 
    deltaUx = Ux2 - np.conj(np.transpose(Ux2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
    tr_deltaUx = np.trace(deltaUx, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
    tr_partx = (tr_deltaUx[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
    Ax = -(1 / (2j * gcoupl * a_perp)) * (deltaUx - tr_partx)
    
    deltaUy = Uy2 - np.conj(np.transpose(Uy2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
    tr_deltaUy = np.trace(deltaUy, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
    tr_party = (tr_deltaUy[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
    Ay = -(1 / (2j * gcoupl * a_perp)) * (deltaUy - tr_party)
    
    deltaUeta = Ueta2 - np.conj(np.transpose(Ueta2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
    tr_deltaUeta = np.trace(deltaUeta, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
    tr_parteta = (tr_deltaUeta[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
    Aeta = -(1 / (2j * gcoupl * a_eta)) * (deltaUeta - tr_parteta)   
        
   
    def project_transverse_fft(Ax, Ay, Aeta, a_perp):
         """
         Proiezione trasversa che rispetta la discretizzazione backward.
         Applica il proiettore trasverso nello spazio dei momenti discreti,
         ai campi in ingresso Ax e Ay.
         Restituisce le trasformate di Fourier di Ax, Ay, Aeta, 
         con Ax e Ay trasversi.
         Le matrici in ingresso a eta fissato, quindi di shape Nx Nx 3 3.
         NOTA: l'output va poi moltiplicato per a_perp^2.
         """
         
         # Rimuove la eventuale traccia dei campi in ingresso
         tr_Ax = np.trace(Ax, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_partx = (tr_Ax[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Ax =  Ax - tr_partx
         
         tr_Ay = np.trace(Ay, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_party = (tr_Ay[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Ay =  Ay - tr_party
         
         tr_Aeta = np.trace(Aeta, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_parteta = (tr_Aeta[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Aeta =  Aeta - tr_parteta
         
         # FFT dei campi
         Akx = fft2(Ax, axes=(0, 1), norm=None)
         Aky = fft2(Ay, axes=(0, 1), norm=None)
     
         Nx, Ny = Ax.shape[:2]
         
         # Momenti discreti (compatibili con differenze finite backward)
         kx = 2 * np.pi * np.fft.fftfreq(Nx, d=a_perp)
         ky = 2 * np.pi * np.fft.fftfreq(Ny, d=a_perp)
         KX, KY = np.meshgrid(kx, ky, indexing='ij')
     
         # Derivate compatibili con differenza finita backward
         KtildeX = np.exp(-1j * KX * a_perp) - 1
         KtildeY = np.exp(-1j * KY * a_perp) - 1
     
         # Norma al quadrato
         Ktilde2 = np.abs(KtildeX)**2 + np.abs(KtildeY)**2
         Ktilde2[Ktilde2 == 0] = 1.0  # evita divisione per zero
     
         # Proiettore trasverso discreto
         Pxx = 1 - (np.abs(KtildeX)**2 / Ktilde2)
         Pxy = - (KtildeX.conj() * KtildeY) / Ktilde2
         Pyx = - (KtildeY.conj() * KtildeX) / Ktilde2
         Pyy = 1 - (np.abs(KtildeY)**2 / Ktilde2)
     
         # Applica il proiettore a ogni componente di colore
         Akx_proj = Pxx[..., None, None] * Akx + Pxy[..., None, None] * Aky
         Aky_proj = Pyx[..., None, None] * Akx + Pyy[..., None, None] * Aky
     
        
         Aketa = fft2(Aeta, axes=(0, 1), norm=None)

         return Akx_proj, Aky_proj, Aketa 
     
    def project_transverse_fft_nopro(Cx, Cy, Ceta, a_perp):
         """
         
         Restituisce le trasformate di Fourier di Ax, Ay, Aeta, 
         senza fare proiezione sul gauge di Coulomb.
         Le matrici in ingresso a eta fissato, quindi di shape Nx Nx 3 3.
         NOTA: la norma delle fft2 deve essere None altrimenti compaiono
         fattori spuri 1/sqrt(Nx Ny) nelle trasformate, che hanno effetto
         sullo spettro.
         NOTA: l'output va poi moltiplicato per a_perp^2.'
         """
         
         # Rimuove la eventuale traccia dei campi in ingresso
         tr_Cx = np.trace(Cx, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_partx = (tr_Cx[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Cx =  Cx - tr_partx
         
         tr_Cy = np.trace(Cy, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_party = (tr_Cy[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Cy =  Cy - tr_party
         
         tr_Ceta = np.trace(Ceta, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
         tr_parteta = (tr_Ceta[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
         Ceta =  Ceta - tr_parteta
         
        
         Cketa = fft2(Ceta, axes=(0, 1), norm=None)
         Ckx_proj = fft2(Cx, axes=(0, 1), norm=None)
         Cky_proj = fft2(Cy, axes=(0, 1), norm=None)
    
         return Ckx_proj, Cky_proj, Cketa         


    Ekx2   = np.empty_like(Ex2)
    Eky2   = np.empty_like(Ey2)
    Eketa2 = np.empty_like(Eeta2)
    Akx2   = np.empty_like(Ex2)
    Aky2   = np.empty_like(Ey2)
    Aketa2 = np.empty_like(Eeta2)

#------------------------------------------------------------------------------
# niter_OR = 0: NO proiezione trasversa discreta
# niter_OR = 1: proiezione trasversa discreta
#------------------------------------------------------------------------------
    
    if niter_OR == 0: 

        Ekx2, Eky2, Eketa2 = project_transverse_fft_nopro(Ex2, Ey2, Eeta2, aperp)
        Ekx2 *= a_perp**2
        Eky2 *= a_perp**2
        Eketa2 *= a_perp**2
        Akx2, Aky2, Aketa2 = project_transverse_fft_nopro(Ax, Ay, Aeta, aperp)
        Akx2 *= a_perp**2
        Aky2 *= a_perp**2
        Aketa2 *= a_perp**2
        
    elif niter_OR == 1:    
        
        Ekx2, Eky2, Eketa2 = project_transverse_fft(Ex2, Ey2, Eeta2, aperp)
        Ekx2 *= a_perp**2
        Eky2 *= a_perp**2
        Eketa2 *= a_perp**2
        Akx2, Aky2, Aketa2 = project_transverse_fft(Ax, Ay, Aeta, aperp)
        Akx2 *= a_perp**2
        Aky2 *= a_perp**2
        Aketa2 *= a_perp**2
    
   
    prod = Ekx2 @ Ekx2.conj().transpose(0, 1, 3, 2)
    tr_Ex2 =  np.trace(prod, axis1=2, axis2=3).real
    prod = Eky2 @ Eky2.conj().transpose(0, 1, 3, 2)
    tr_Ey2 =  np.trace(prod, axis1=2, axis2=3).real
    prod = Eketa2 @ Eketa2.conj().transpose(0, 1, 3, 2)
    tr_Eeta2 =  np.trace(prod, axis1=2, axis2=3).real
    
    prod = Akx2 @ Akx2.conj().transpose(0, 1, 3, 2)
    tr_Ax2 =  np.trace(prod, axis1=2, axis2=3).real
    prod = Aky2 @ Aky2.conj().transpose(0, 1, 3, 2)
    tr_Ay2 =  np.trace(prod, axis1=2, axis2=3).real
    prod = Aketa2 @ Aketa2.conj().transpose(0, 1, 3, 2)
    tr_Aeta2 =  np.trace(prod, axis1=2, axis2=3).real
    
    prod = (Ekx2 @ Akx2.conj().transpose(0, 1, 3, 2) 
            - Akx2 @ Ekx2.conj().transpose(0, 1, 3, 2) )
    tr_ExAx =  1j *np.trace(prod, axis1=2, axis2=3)
    prod =   (Eky2 @ Aky2.conj().transpose(0, 1, 3, 2) 
            - Aky2 @ Eky2.conj().transpose(0, 1, 3, 2) )
    tr_EyAy = 1j *  np.trace(prod, axis1=2, axis2=3)
    prod = (Eketa2 @ Aketa2.conj().transpose(0, 1, 3, 2) 
            - Aketa2 @ Eketa2.conj().transpose(0, 1, 3, 2) )
    tr_EetaAeta = 1j * np.trace(prod, axis1=2, axis2=3)
    
    tr_ODD = (tr_ExAx + tr_EyAy + tr_EetaAeta).real

    

    # dNd2k_FULL = np.zeros((Nx, Ny))
    # dNd2k_noODD = np.zeros_like(dNd2k_FULL)
    # dNd2k_EQUIPA = np.zeros_like(dNd2k_FULL)   
            
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=a_perp)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=a_perp)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    mglueGEV = 0.0002 
    mglueFM = mglueGEV / hbarc
    
    KtildeX = np.sin(0.5 * KX * aperp)
    KtildeY = np.sin(0.5 * KY * aperp)
    KmodSQ = (4.0 / aperp**2) * (KtildeX**2 + KtildeY**2)   
    Kmod = np.sqrt(KmodSQ + mglueFM**2)  # aggiunge massa infrarossa

    
   


    
    
    dNd2k_1 =    (tr_Ex2 + tr_Ey2)/ (proptime * Kmod)
    dNd2k_2 =   (tr_Ax2 + tr_Ay2) * proptime * Kmod 
    dNd2k_3 =    (tr_Eeta2 * proptime) / Kmod
    dNd2k_4 =    (tr_Aeta2 * Kmod) / proptime
    dNd2K_ODD = tr_ODD
                                      
    prefactor = 1 / (2 * np.pi * hbarc)**2
    
   
    
    dNd2k_FULL = prefactor * ( dNd2k_1 + dNd2k_2 + dNd2k_3 + dNd2k_4 
                             + dNd2K_ODD)
        
       
    dNd2k_noODD = prefactor * ( dNd2k_1 + dNd2k_2 + dNd2k_3 + dNd2k_4 )
        
       
    dNd2k_EQUIPA = 2.0 * prefactor * ( dNd2k_1 + dNd2k_3)
    
    
   
    
    

#------------------------------------------------------------------------------ 
# KX e KY sono convertiti in GeV per la scrittura sul file
#------------------------------------------------------------------------------   
    # KX = hbarc * np.fft.fftshift(KX)  
    # KY = hbarc * np.fft.fftshift(KY)
    
    KtildeX*= hbarc * 2.0 / aperp
    KtildeY*= hbarc * 2.0 / aperp
    
    KtildeX = np.fft.fftshift(KtildeX)  
    KtildeY = np.fft.fftshift(KtildeY)
    
    dNd2k_FULL = np.fft.fftshift(dNd2k_FULL)
    dNd2k_noODD = np.fft.fftshift(dNd2k_noODD)
    dNd2k_EQUIPA = np.fft.fftshift(dNd2k_EQUIPA)

    

    os.makedirs("spectrum_event", exist_ok=True)
    with open(f"spectrum_event/spectrum_event{event_id}.csv", "w") as f:
        for ix in range(Nx):
            for iy in range(Ny):
                f.write(f"{KtildeX[ix, iy]:.6e}, {KtildeY[ix, iy]:.6e}, \
                        {dNd2k_FULL[ix, iy]:.6e},\
                        {dNd2k_noODD[ix, iy]:.6e}, \
                        {dNd2k_EQUIPA[ix, iy]:.6e}\n")



    # return dNd2k

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    

    
def print_rho_stats(rho, a=1):
    """
    Stampa statistiche (max, min, mean, var) della componente rho^a(x, y).

    Parametri:
    - rho: array shape (Nc², Nx, Ny), con rho[0] = 0 (padding).
    - a: indice della componente colore da analizzare (1-8 per SU(3)).
    """
    if a < 1 or a >= rho.shape[0]:
        raise ValueError(f"Indice colore a={a} non valido. Deve essere in [1, {rho.shape[0]-1}].")

    comp = rho[a]  # Seleziona la componente a
    
    max_val = np.max(comp)
    min_val = np.min(comp)
    mean_val = np.mean(comp)
    var_val = np.var(comp)
    var_comp = (g * mu_A_fm)**2  / (a_perp**2 * Nsheets)

    print(f"Statistiche di rho^{a}(x,y):")
    print(f"  Max = {max_val}")
    print(f"  Min = {min_val}")
    print(f"  Mean = {mean_val}")
    print(f"  Var = {var_val}\n")
    print(f"  VarComp = {var_comp}\n")
    
    
    
    
#------------------------------------------------------------------------------
# FUNZIONI PER GLI HEAVY QUARKS
#------------------------------------------------------------------------------

def initialize_HQ_run(nrun):


    


    # 1. Costruisci i nomi dei due file
    file1 = f"HQinitializations/hq_init_event{nrun}_proj1.csv"
    file2 = f"HQinitializations/hq_init_event{nrun}_proj2.csv"

    # 2. Leggi i due file CSV
    data1 = np.loadtxt(file1, delimiter=",")  # colonne: x y gmu_norm
    data2 = np.loadtxt(file2, delimiter=",")  # stesso formato

#------------------------------------------------------------------------------
# COORDINATES INIZIALITATION    
#------------------------------------------------------------------------------

    # 3. Estrai colonne (x, y sono uguali nei due)
    x = data1[:, 0]
    y = data1[:, 1]
    gmu1 = data1[:, 2]
    gmu2 = data2[:, 2]

    # 4. Calcola la densità normalizzata
    density = gmu1 * gmu2
    # assert np.all(density >= 0), f"Density contiene valori negativi! Minimo: {np.min(density)}"
    density /= np.sum(density)  # normalizza (sommata fa 1)

    # 5. Estrai NHQ punti secondo la densità
    Nhalf = NHQ // 2
    indices = np.random.choice(len(density), size=Nhalf, replace=True, p=density)
    XHQ_q = x[indices]
    YHQ_q = y[indices]

    # Smearing continuo: sposta ciascun HQ in un quadratino centrato sulla cella
    XHQ_q += (np.random.rand(Nhalf) - 0.25) * a_perp
    YHQ_q += (np.random.rand(Nhalf) - 0.25) * a_perp
    
    
    sigma_corr = cfg.sigma_corr
    # Coord antiquark: gaussiane intorno ai rispettivi quark
    XHQ_a = XHQ_q + np.random.normal(loc=0.0, scale=sigma_corr, size=Nhalf)
    YHQ_a = YHQ_q + np.random.normal(loc=0.0, scale=sigma_corr, size=Nhalf)
    
    
    XHQ = np.concatenate((XHQ_q, XHQ_a))
    YHQ = np.concatenate((YHQ_q, YHQ_a))
    
    # Inizializzazione coordinata eta per ogni HQ
    if i_fluct == 0:
        etaHQ = None
    else:
        etaHQ = np.random.uniform(-Leta/2, Leta/2, size=NHQ)

    # Applica periodic boundary conditions (PBC) sul box centrato
    XHQ = (XHQ + Lx/2) % Lx - Lx/2
    YHQ = (YHQ + Ly/2) % Ly - Ly/2

    if plot_hqs.strip().lower() == "yes":
        # Distanze tra quark e antiquark (distanze!)
        X_q = XHQ[:Nhalf]
        Y_q = YHQ[:Nhalf]
        X_a = XHQ[Nhalf:]
        Y_a = YHQ[Nhalf:]
        
        dx = X_q - X_a
        dy = Y_q - Y_a
        
        # Applica periodic boundary conditions alla distanza
        dx = (dx + Lx/2) % Lx - Lx/2
        dy = (dy + Ly/2) % Ly - Ly/2
        
        distanza_T = np.sqrt(dx**2 + dy**2)
        
        df_dist = pd.DataFrame({'dist_quark_antiquark [fm]': distanza_T})
        csv_filename = f"HQinitializations/distanze_event{nrun}.csv"
        df_dist.to_csv(csv_filename, index=False)
        
        ratios_T = distanza_T / sigma_corr
        plt.figure(figsize=(6,4))
        plt.hist(ratios_T, bins=40, color='orange', edgecolor='black', alpha=0.8)
        plt.xlabel(r"Distanza quark-antiquark/$\sigma$ ")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione della distanza_T")
        plt.tight_layout()
        plt.savefig(f"HQinitializations/distanzaT_hist_event{nrun}.png", dpi=300)


    
    # import matplotlib.pyplot as plt

    def plot_density_with_HQs(title="HQ sampling"):
        # Reshape per creare la mappa 2D (griglia)
        X = x.reshape((Nx, Ny))
        Y = y.reshape((Nx, Ny))
        Z = density.reshape((Nx, Ny))
    
        plt.figure(figsize=(6, 5))
        
        
        zmin = Z.min()
        zmax = Z.max()
        
        if np.isclose(zmin, zmax):
            zmax += 1e-6  # evita errore matplotlib se mappa piatta
        
        levels = np.linspace(zmin, zmax, 50)
        
        cp = plt.contourf(X, Y, Z, levels=levels, cmap='viridis', vmin=0)

         

        plt.colorbar(cp, label=r'Densità $g\mu_1 \cdot g\mu_2$')
        plt.scatter(XHQ, YHQ, color='red', s=10, label='HQs')
        plt.xlabel("x [fm]")
        plt.ylabel("y [fm]")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        filename = f'HQinitializations/HQs_coord_evento_{nrun}.png'
        plt.savefig(filename, dpi=300)

        # plt.show()

    
    if plot_hqs.strip().lower() == "yes":
        plot_density_with_HQs()
        
        
#------------------------------------------------------------------------------
# MOMENTA INIZIALITATION    
#------------------------------------------------------------------------------
    def sample_fonll_pt_and_momenta():
        """
        Estrae NHQ impulsi trasversi (p_x, p_y) secondo la distribuzione FONLL.
    
        Parametri:
        - NHQ: numero di quark pesanti da generare
        - flavor: "charm" o "beauty"
    
        Ritorna:
        - PXHQ: array (NHQ,) componenti x degli impulsi
        - PYHQ: array (NHQ,) componenti y degli impulsi
        """
    
        # Parametri FONLL per charm e beauty
        if hq_flavor.lower() == "charm":
            x0 = 20.2837
            x1 = 1.95061
            x2 = 3.13695
            x3 = 0.075166
        elif hq_flavor.lower() == "beauty":
            x0 = 0.46799
            x1 = 1.83805
            x2 = 3.07569
            x3 = 0.030156
        else:
            raise ValueError("Flavor non valido. Usa 'charm' o 'beauty'.")
    
        # Funzione di densità non normalizzata
        def fonll_pt_dist(pt):
            return pt * x0 / (1 + x3 * pt**x1)**x2
    
        # Sampling con accettazione-rifiuto
        pT_min = 0.01   
        pT_max = 10.0   
        
        pt_vals = np.linspace(pT_min, pT_max, 1000)
        f_max = fonll_pt_dist(pt_vals).max()
    
        Nhalf = NHQ // 2   
        sampled = []
        while len(sampled) < Nhalf:
            pt_trial = np.random.uniform(pT_min, pT_max)
            y = np.random.uniform(0, f_max)
            if y < fonll_pt_dist(pt_trial):
                sampled.append(pt_trial)
    
        pT = np.array(sampled[:Nhalf])
    
        # Estrai angolo random tra 0 e 2π
        phi = np.random.uniform(0, 2 * np.pi, size=Nhalf)
    
        PXHQ_q = pT * np.cos(phi)
        PYHQ_q = pT * np.sin(phi)
        
        PXHQ_a = -PXHQ_q
        PYHQ_a = -PYHQ_q
        
        PXHQ = np.concatenate((PXHQ_q, PXHQ_a))
        PYHQ = np.concatenate((PYHQ_q, PYHQ_a))

    
        return PXHQ, PYHQ
    
    if init_mom_space == "FONLL":
        PXHQ, PYHQ = sample_fonll_pt_and_momenta()    # momenta in GeV
        
        
        
    def plot_mom_density_with_HQs(title=r"HQ $p_T$ sampling"):
        # Reshape per creare la mappa 2D (griglia)
        
         
    
        plt.figure(figsize=(6, 5))
         
        plt.scatter(PXHQ, PYHQ, color='red', s=10, label='HQs')
        plt.xlabel(r"$p_x$ [GeV]")
        plt.ylabel(r"$p_y$ [GeV]")
        
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        filename = f'HQinitializations/HQs_coordP_evento_{nrun}.png'
        plt.savefig(filename, dpi=300)

        # plt.show()

    
    if plot_hqs.strip().lower() == "yes":
        plot_mom_density_with_HQs()    
        
        
        
    PXHQ /= hbarc                                 # momenta in 1/fm
    PYHQ /= hbarc

    """Momenta are returned in 1/fm to the main code, in order to match
    the physical dimensions of the color-electric fields in the CYM 
    evolution"""
    
#------------------------------------------------------------------------------
# COLOR CHARGES INIZIALITATION    
#------------------------------------------------------------------------------
    
    def random_su3():
        Z = (np.random.randn(3,3) + 1j * np.random.randn(3,3)) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        D = np.diag(np.exp(1j * np.angle(np.diag(R))))
        U = Q @ D
        return U / np.linalg.det(U)**(1/3)
    
    
    # Vettore iniziale Q^a_0 (solo componenti 5 e 8)
    Q0_vec = np.zeros(8)
    Q0_vec[4] = -1.69469   # Q0^5
    Q0_vec[7] = -1.06209   # Q0^8

    T_a = get_su3_generators()  # shape (8, 3, 3)
    
    Q0 = sum(Q0_vec[a] * T_a[a] for a in range(8))  # shape (3, 3)
    
    Qmat = np.array([U.conj().T @ Q0 @ U for U in \
                     [random_su3() for _ in range(NHQ)]])  # shape (NHQ, 3, 3)
        
    PetaHQ=np.zeros_like(PXHQ, dtype=np.float64)
   

    return XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, Qmat
    

def evolve_X_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, mHQ, prop_time, delta_tau, i_fluct):
#PetaHQ,

    """
    Evolvi le coordinate XHQ e YHQ degli heavy quarks
    al tempo tau + delta_tau/2 in uno schema leapfrog.

    Parametri:
    - XHQ, YHQ: array (NHQ,) delle coordinate attuali
    - PXHQ, PYHQ: array (NHQ,) degli impulsi (in 1/fm)
    - tau: tempo proprio attuale (in fm)
    - delta_tau: passo temporale (in fm)

    Ritorna:
    - XHQ_new, YHQ_new: array (NHQ,) con le coordinate aggiornate
    """    
    # Compute velocity: v^i = p^i / E, where E = sqrt(p^2 + m^2)
    # Per ora assumiamo m costante globale
    p_squared = PXHQ**2 + PYHQ**2  + (prop_time*PetaHQ)**2
    E = np.sqrt(p_squared + mHQ**2)  # mass_HQ globale o passato come argomento

    v_x = PXHQ / E
    v_y = PYHQ / E
    
    # Leapfrog: aggiorna coordinate a τ + Δτ/2
    XHQ_new = XHQ + v_x * delta_tau
    YHQ_new = YHQ + v_y * delta_tau
    
    # Periodic Boundary Conditions 
    XHQ_new = (XHQ_new + Lx/2) % Lx - Lx/2
    YHQ_new = (YHQ_new + Ly/2) % Ly - Ly/2
    
    etaHQ_new = etaHQ

    if i_fluct !=0:
        v_eta = PetaHQ / E
        etaHQ_new = etaHQ + v_eta * delta_tau
        etaHQ_new = (etaHQ_new + Leta/2) % Leta - Leta/2 

    return XHQ_new, YHQ_new, etaHQ_new

def evolve_P_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, QHQ, Ex, Ey, Eeta, Ux, Uy, Ueta, mHQ, prop_tau, delta_tau, i_fluct):
#PetaHQ,
    
    """
    Evolve momenta (PXHQ, PYHQ, PetaHQ) of heavy quarks using color-electric fields 
    and gauge links.  
    """
    # PetaHQ=np.zeros_like(PXHQ, dtype=np.float64)
    if i_fluct == 0:
        neta_slice = (N_eta -1) // 2

        p_squared = PXHQ**2 + PYHQ**2 + (prop_tau*PetaHQ)**2
        E = np.sqrt(p_squared + mHQ**2)  # mass_HQ globale o passato come argomento

        local_Uxy=np.zeros_like(QHQ, dtype=np.complex128)
        local_Uetax=np.zeros_like(QHQ, dtype=np.complex128)
        local_Uetay=np.zeros_like(QHQ, dtype=np.complex128)

    

        local_Ex = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Ey = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Eeta = np.zeros((NHQ, 3, 3), dtype=np.complex128)

        identity_NHQ = np.array([np.eye(3) for _ in range(NHQ)])

        Uxy, Uetax, Uetay = calculate_plaquette_perB(Ux, Uy, Ueta)

        for hq in range(NHQ):
            xi_hq = np.floor(XHQ[hq] / a_perp).astype(int) + Nx // 2
            yi_hq = np.floor(YHQ[hq] / a_perp).astype(int) + Nx // 2
            xi_hq = 0 if xi_hq < 0 else (Nx-1 if xi_hq >= Nx else xi_hq) # CLAMP
            yi_hq = 0 if yi_hq < 0 else (Nx-1 if yi_hq >= Nx else yi_hq) # CLAMP
            
            local_Uxy[hq]= Uxy[xi_hq, yi_hq, neta_slice]
            local_Uetax[hq] = Uetax[xi_hq, yi_hq, neta_slice]
            local_Uetay[hq] = Uetay[xi_hq, yi_hq, neta_slice]
            local_Ex[hq] = Ex[xi_hq, yi_hq, neta_slice]
            local_Ey[hq] = Ey[xi_hq, yi_hq, neta_slice]
            local_Eeta[hq] = Eeta[xi_hq, yi_hq, neta_slice]

        AH_Uetay = 0.5 * (local_Uetay - np.conj(np.swapaxes(local_Uetay, 1, 2)))
        AH_Uetax = 0.5 * (local_Uetax - np.conj(np.swapaxes(local_Uetax, 1, 2)))
        AH_Uxy   = 0.5 * (local_Uxy   - np.conj(np.swapaxes(local_Uxy,   1, 2)))
        tr_AH_Uetay = (np.einsum('nii->n', AH_Uetay) / 3.0)[:, None, None]
        tr_AH_Uetax = (np.einsum('nii->n', AH_Uetax) / 3.0)[:, None, None]
        tr_AH_Uxy   = (np.einsum('nii->n', AH_Uxy  ) / 3.0)[:, None, None]
    
        proj_Uetay = AH_Uetay - tr_AH_Uetay * identity_NHQ
        proj_Uetax = AH_Uetax - tr_AH_Uetax * identity_NHQ
        proj_Uxy   = AH_Uxy   - tr_AH_Uxy   * identity_NHQ

        #----------------------------------------------------------------------------
        #  CHECKED COEFFICIENTs on 06 nov 2026
        coef_x = -1.0 / (1j * g * a_eta * a_perp * prop_tau)
        coef_y = +1.0 / (1j * g * a_eta * a_perp * prop_tau)
        coef_e = +1.0 / (1j * g * (a_perp**2))
    
        local_Bx   = coef_x * proj_Uetay   # B_x = +F_{ηy}/τ
        local_By   = coef_y * proj_Uetax   # B_y = -F_{ηx}/τ
        local_Beta = coef_e * proj_Uxy     # B_η = -F_{xy}

        # F_{munu} = -1/2iga_mu a_nu (U_{munu} - U_{munu}† - 1/3 Tr(U_{munu} - U_{munu}†) I)
        # according to the gauge links choice of this code, U=exp(-iga A), quindi U_munu = exp(-iga_mu a_nu F_{munu}) + O(a^3)
        #----------------------------------------------------------------------------


        
    
        #checked
        PXHQ_new=PXHQ+delta_tau*((g/(1/2))*((1/prop_tau)*np.real(np.einsum('ijk,ikj->i', QHQ, local_Ex))   +np.real(np.einsum('ijk,ikj->i', QHQ, local_Beta))*(PYHQ/E)  -prop_tau*np.real(np.einsum('ijk,ikj->i', QHQ, local_By))*(PetaHQ/E))) 

        #checked
        PYHQ_new=PYHQ+delta_tau*((g/(1/2))*((1/prop_tau)*np.real(np.einsum('ijk,ikj->i', QHQ, local_Ey))       -np.real(np.einsum('ijk,ikj->i', QHQ, local_Beta))*(PXHQ/E)   +prop_tau*np.real(np.einsum('ijk,ikj->i', QHQ, local_Bx))*(PetaHQ/E)))  

        PrefPeta = 2 * g * delta_tau * prop_tau / (prop_tau + delta_tau )**2 # il 2 overall viene da 1/TR con TR = 1/2
        
        p1Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_Eeta)) 
        p2Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_Bx)) *(PYHQ/E) 
        p3Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_By))*(PXHQ/E)

        PetaHQ_new = PetaHQ * (prop_tau  / (prop_tau + delta_tau) )**2 + PrefPeta * (p1Peta -p2Peta + p3Peta)
        # questo sopra è codice giusto


        
    #    PXHQ_new=PXHQ
    #    PYHQ_new=PYHQ
    #    PetaHQ_new=PetaHQ

        return PXHQ_new, PYHQ_new, PetaHQ_new
    
    else:
        p_squared = PXHQ**2 + PYHQ**2 + (prop_tau*PetaHQ)**2
        E = np.sqrt(p_squared + mHQ**2)  # mass_HQ globale o passato come argomento

        local_Uxy=np.zeros_like(QHQ, dtype=np.complex128)
        local_Uetax=np.zeros_like(QHQ, dtype=np.complex128)
        local_Uetay=np.zeros_like(QHQ, dtype=np.complex128)

    

        local_Ex = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Ey = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Eeta = np.zeros((NHQ, 3, 3), dtype=np.complex128)

        identity_NHQ = np.array([np.eye(3) for _ in range(NHQ)])

        Uxy, Uetax, Uetay = calculate_plaquette_perB(Ux, Uy, Ueta)

        for hq in range(NHQ):
            xi_hq = np.floor(XHQ[hq] / a_perp).astype(int) + Nx // 2
            yi_hq = np.floor(YHQ[hq] / a_perp).astype(int) + Nx // 2
            etai_hq = np.floor(etaHQ[hq] / a_eta).astype(int) + N_eta // 2
            xi_hq = 0 if xi_hq < 0 else (Nx-1 if xi_hq >= Nx else xi_hq) # CLAMP
            yi_hq = 0 if yi_hq < 0 else (Nx-1 if yi_hq >= Nx else yi_hq) # CLAMP
            etai_hq = 0 if etai_hq < 0 else (N_eta-1 if etai_hq >= N_eta else etai_hq) # CLAMP
            
            # local_Uxy[hq]= Uxy[xi_hq, yi_hq, 0]
            # local_Uetax[hq] = Uetax[xi_hq, yi_hq, 0]
            # local_Uetay[hq] = Uetay[xi_hq, yi_hq, 0]
            # local_Ex[hq] = Ex[xi_hq, yi_hq, 0]
            # local_Ey[hq] = Ey[xi_hq, yi_hq, 0]
            # local_Eeta[hq] = Eeta[xi_hq, yi_hq, 0]

            local_Uxy[hq]= Uxy[xi_hq, yi_hq, etai_hq]
            local_Uetax[hq] = Uetax[xi_hq, yi_hq, etai_hq]
            local_Uetay[hq] = Uetay[xi_hq, yi_hq, etai_hq]
            local_Ex[hq] = Ex[xi_hq, yi_hq, etai_hq]
            local_Ey[hq] = Ey[xi_hq, yi_hq, etai_hq]
            local_Eeta[hq] = Eeta[xi_hq, yi_hq, etai_hq]


        AH_Uetay = 0.5 * (local_Uetay - np.conj(np.swapaxes(local_Uetay, 1, 2)))
        AH_Uetax = 0.5 * (local_Uetax - np.conj(np.swapaxes(local_Uetax, 1, 2)))
        AH_Uxy   = 0.5 * (local_Uxy   - np.conj(np.swapaxes(local_Uxy,   1, 2)))
        tr_AH_Uetay = (np.einsum('nii->n', AH_Uetay) / 3.0)[:, None, None]
        tr_AH_Uetax = (np.einsum('nii->n', AH_Uetax) / 3.0)[:, None, None]
        tr_AH_Uxy   = (np.einsum('nii->n', AH_Uxy  ) / 3.0)[:, None, None]
    
        proj_Uetay = AH_Uetay - tr_AH_Uetay * identity_NHQ
        proj_Uetax = AH_Uetax - tr_AH_Uetax * identity_NHQ
        proj_Uxy   = AH_Uxy   - tr_AH_Uxy   * identity_NHQ
    
        
        #----------------------------------------------------------------------------
        #  CHECKED COEFFICIENTs on 06 nov 2026
        coef_x = -1.0 / (1j * g * a_eta * a_perp * prop_tau)
        coef_y = +1.0 / (1j * g * a_eta * a_perp * prop_tau)
        coef_e = +1.0 / (1j * g * (a_perp**2))
    
        local_Bx   = coef_x * proj_Uetay   # B_x = +F_{ηy}/τ
        local_By   = coef_y * proj_Uetax   # B_y = -F_{ηx}/τ
        local_Beta = coef_e * proj_Uxy     # B_η = -F_{xy}

        # F_{munu} = -1/2iga_mu a_nu (U_{munu} - U_{munu}† - 1/3 Tr(U_{munu} - U_{munu}†) I)
        # according to the gauge links choice of this code, U=exp(-iga A), quindi U_munu = exp(-iga_mu a_nu F_{munu}) + O(a^3)
        #----------------------------------------------------------------------------




        
    

        PXHQ_new=PXHQ+delta_tau*((g/(1/2))*((1/prop_tau)*np.real(np.einsum('ijk,ikj->i', QHQ, local_Ex))   +np.real(np.einsum('ijk,ikj->i', QHQ, local_Beta))*(PYHQ/E)  -prop_tau*np.real(np.einsum('ijk,ikj->i', QHQ, local_By))*(PetaHQ/E))) 

        PYHQ_new=PYHQ+delta_tau*((g/(1/2))*((1/prop_tau)*np.real(np.einsum('ijk,ikj->i', QHQ, local_Ey))       -np.real(np.einsum('ijk,ikj->i', QHQ, local_Beta))*(PXHQ/E)   +prop_tau*np.real(np.einsum('ijk,ikj->i', QHQ, local_Bx))*(PetaHQ/E)))  

        PrefPeta = 2 * g * delta_tau * prop_tau / (prop_tau + delta_tau )**2 # il 2 overall viene da 1/TR con TR = 1/2
        
        p1Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_Eeta)) 
        p2Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_Bx)) *(PYHQ/E) 
        p3Peta = np.real(np.einsum('ijk,ikj->i', QHQ, local_By))*(PXHQ/E)

        PetaHQ_new = PetaHQ * (prop_tau  / (prop_tau + delta_tau) )**2 + PrefPeta * (p1Peta -p2Peta + p3Peta)

        
    #    PXHQ_new=PXHQ
    #    PYHQ_new=PYHQ
    #    PetaHQ_new=PetaHQ

        return PXHQ_new, PYHQ_new, PetaHQ_new



def evolve_Q_hq(XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, QHQ, Ex, Ey, Eeta, Ux, Uy, Ueta, mHQ, prop_tau, delta_tau, i_fluct):
    
    """
    Evolve momenta (PXHQ, PYHQ, PetaHQ) of heavy quarks using color-electric fields 
    and gauge links.  
    """
    if i_fluct==0:   #CODICE CON SLICE DI RAPIDITA' FISSATA E NON-EVOLVING CON WONG EQ. (per sistema boost-invariante, privo di fluttuazioni sul bulk)
        # PetaHQ=np.zeros_like(PXHQ, dtype=np.float64)
    
        gcoupl = g
                
        neta_slice = (N_eta - 1) // 2
        
        Ux2 = Ux[:, :, neta_slice, :, :]
        Uy2 = Uy[:, :, neta_slice, :, :]
        Ueta2 = Ueta[:, :, neta_slice, :, :]

        p_squared = PXHQ**2 + PYHQ**2 + (prop_tau*PetaHQ)**2
        E = np.sqrt(p_squared + mHQ**2)  # mass_HQ globale o passato come argomento
        
        
        # local_Ux = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_Uy = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_Ueta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        
        # identity_NHQ = np.array([np.eye(3) for _ in range(NHQ)])
        
        Ax = np.zeros((Nx, Ny, 3, 3), dtype=complex)
        Ay = np.zeros((Nx, Ny, 3, 3), dtype=complex)
        Aeta = np.zeros((Nx, Ny, 3, 3), dtype=complex)
        
        deltaUx = Ux2 - np.conj(np.transpose(Ux2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
        tr_deltaUx = np.trace(deltaUx, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
        tr_partx = (tr_deltaUx[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
        Ax = -(1 / (2j * gcoupl * a_perp)) * (deltaUx - tr_partx)

        deltaUy = Uy2 - np.conj(np.transpose(Uy2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
        tr_deltaUy = np.trace(deltaUy, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
        tr_party = (tr_deltaUy[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
        Ay = -(1 / (2j * gcoupl * a_perp)) * (deltaUy - tr_party)
        
        deltaUeta = Ueta2 - np.conj(np.transpose(Ueta2, axes=(0, 1, 3, 2)))  # U - U†, shape (Nx, Ny, 3, 3)
        tr_deltaUeta = np.trace(deltaUeta, axis1=2, axis2=3)               # Tr(U - U†), shape (Nx, Ny)
        tr_parteta = (tr_deltaUeta[..., None, None] / 3.0) * np.eye(3)[None, None, :, :]  # shape (Nx, Ny, 3, 3)
        Aeta = -(1 / (2j * gcoupl * a_eta)) * (deltaUeta - tr_parteta)


        local_Ax = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Ay = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Aeta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_x = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_y = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_eta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        
        
        for hq in range(NHQ):
            xi_hq = np.floor(XHQ[hq] / a_perp).astype(int) + Nx // 2
            yi_hq = np.floor(YHQ[hq] / a_perp).astype(int) + Nx // 2
            
            xi_hq = np.clip(xi_hq, 0, Nx-1)     # CLAMP
            yi_hq = np.clip(yi_hq, 0, Nx-1)

            
            local_Ax[hq] = Ax[xi_hq, yi_hq]
            local_Ay[hq] = Ay[xi_hq, yi_hq]
            local_Aeta[hq] = Aeta[xi_hq, yi_hq]
            
    
        
        # local_comm_x = np.einsum('nij,njk->nik', QHQ, local_Ax) - np.einsum('nij,njk->nik', local_Ax, QHQ)
        # local_comm_y = np.einsum('nij,njk->nik', QHQ, local_Ay) - np.einsum('nij,njk->nik', local_Ay, QHQ)     
        # 
        #    
    # velocità degli HQs    
        ratio_x = (PXHQ / E)[:, None, None]   # espande in (NHQ,1,1)
        ratio_y = (PYHQ / E)[:, None, None]
        ratio_eta = (PetaHQ / E)[:, None, None]
        
        
    # A · v = Ax * vx + Ay * vy (shape NHQ x 3 x 3)
        local_A_dot_v = gcoupl*(local_Ax * ratio_x + local_Ay * ratio_y + local_Aeta * ratio_eta)

    # Costruzione di U = exp(i A·v Δτ)
        # Umat = np.array([expm(1j * delta_tau * local_A_dot_v[hq]) for hq in range(NHQ)])
        # Udag = np.conj(np.transpose(Umat, axes=(0, 2, 1)))
    # Costruzione di U = exp(i A·v Δτ)    
    # Usa esponenziazione batch ottimizzata definita nel codice principale
        Umat = expm_su3_batch(-1j * delta_tau * local_A_dot_v)
        Udag = np.conj(np.transpose(Umat, axes=(0, 2, 1)))
        
    # Evoluzione unitaria: Q → U† Q U
        QHQ_new = np.einsum('nij,njk,nkl->nil', Udag, QHQ, Umat)

        
        
        return QHQ_new
    
    else:
        # PetaHQ=np.zeros_like(PXHQ, dtype=np.float64)
        
        gcoupl = g
        

        Ux2 = Ux
        Uy2 = Uy
        Ueta2 = Ueta
        

        p_squared = PXHQ**2 + PYHQ**2 + (prop_tau*PetaHQ)**2
        E = np.sqrt(p_squared + mHQ**2)  # mass_HQ globale o passato come argomento
        
        
        # local_Ux = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_Uy = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_Ueta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        
        # identity_NHQ = np.array([np.eye(3) for _ in range(NHQ)])
        
        Ax = np.zeros((Nx, Ny, N_eta, 3, 3), dtype=complex)
        Ay = np.zeros((Nx, Ny, N_eta, 3, 3), dtype=complex)
        Aeta = np.zeros((Nx, Ny, N_eta, 3, 3), dtype=complex)
        
        deltaUx = Ux2 - np.conj(np.transpose(Ux2, axes=(0, 1, 2, 4, 3)))  # U - U†, shape (Nx, Ny, N_eta, 3, 3)
        tr_deltaUx = np.trace(deltaUx, axis1=3, axis2=4)               # Tr(U - U†), shape (Nx, Ny, N_eta)
        tr_partx = (tr_deltaUx[..., None, None] / 3.0) * np.eye(3)[None, None, None, :, :]  # shape (Nx, Ny, N_eta, 3, 3)
        Ax = -(1 / (2j * gcoupl * a_perp)) * (deltaUx - tr_partx)
       
        deltaUy = Uy2 - np.conj(np.transpose(Uy2, axes=(0, 1, 2, 4, 3)))  # U - U†, shape (Nx, Ny, N_eta, 3, 3)
        tr_deltaUy = np.trace(deltaUy, axis1=3, axis2=4)               # Tr(U - U†), shape (Nx, Ny, N_eta)
        tr_party = (tr_deltaUy[..., None, None] / 3.0) * np.eye(3)[None, None, None, :, :]  # shape (Nx, Ny, N_eta, 3, 3)
        Ay = -(1 / (2j * gcoupl * a_perp)) * (deltaUy - tr_party)
        
        deltaUeta = Ueta2 - np.conj(np.transpose(Ueta2, axes=(0, 1, 2, 4, 3)))  # U - U†, shape (Nx, Ny, N_eta, 3, 3)
        tr_deltaUeta = np.trace(deltaUeta, axis1=3, axis2=4)               # Tr(U - U†), shape (Nx, Ny, N_eta)
        tr_parteta = (tr_deltaUeta[..., None, None] / 3.0) * np.eye(3)[None, None, None, :, :]  # shape (Nx, Ny, N_eta, 3, 3)
        Aeta = -(1 / (2j * gcoupl * a_eta)) * (deltaUeta - tr_parteta)
        
        local_Ax = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Ay = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        local_Aeta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_x = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_y = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        # local_comm_eta = np.zeros((NHQ, 3, 3), dtype=np.complex128)
        
        
        for hq in range(NHQ):
            xi_hq = np.floor(XHQ[hq] / a_perp).astype(int) + Nx // 2
            yi_hq = np.floor(YHQ[hq] / a_perp).astype(int) + Nx // 2
            etai_hq = np.floor(etaHQ[hq] / a_eta).astype(int) + N_eta // 2

            xi_hq = np.clip(xi_hq, 0, Nx-1)     # CLAMP
            yi_hq = np.clip(yi_hq, 0, Nx-1)
            etai_hq = np.clip(etai_hq, 0, N_eta-1)
            
            local_Ax[hq] = Ax[xi_hq, yi_hq, etai_hq]
            local_Ay[hq] = Ay[xi_hq, yi_hq, etai_hq]
            local_Aeta[hq] = Aeta[xi_hq, yi_hq, etai_hq]
            
    
        
        # local_comm_x = np.einsum('nij,njk->nik', QHQ, local_Ax) - np.einsum('nij,njk->nik', local_Ax, QHQ)
        # local_comm_y = np.einsum('nij,njk->nik', QHQ, local_Ay) - np.einsum('nij,njk->nik', local_Ay, QHQ)
    
 #-       
        
        
    # velocità degli HQs    
        ratio_x = (PXHQ / E)[:, None, None]   # espande in (NHQ,1,1)
        ratio_y = (PYHQ / E)[:, None, None]
        ratio_eta = (PetaHQ / E)[:, None, None]
        
        
    # A · v = Ax * vx + Ay * vy (shape NHQ x 3 x 3)
        local_A_dot_v = gcoupl*(local_Ax * ratio_x + local_Ay * ratio_y + local_Aeta * ratio_eta)
        
    # Costruzione di U = exp(i A·v Δτ)
        # Umat = np.array([expm(1j * delta_tau * local_A_dot_v[hq]) for hq in range(NHQ)])
        # Udag = np.conj(np.transpose(Umat, axes=(0, 2, 1)))
    # Costruzione di U = exp(i A·v Δτ)    
    # Usa esponenziazione batch ottimizzata definita nel codice principale
        Umat = expm_su3_batch(-1j * delta_tau * local_A_dot_v)
        Udag = np.conj(np.transpose(Umat, axes=(0, 2, 1)))
        
    # Evoluzione unitaria: Q → U† Q U
        QHQ_new = np.einsum('nij,njk,nkl->nil', Udag, QHQ, Umat)

        
        
        return QHQ_new

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# if len(sys.argv) == 3:
#     nrun = int(sys.argv[1])
#     NRUNTOT = int(sys.argv[2])
# else:
#     print("⚠️ Nessun argomento passato, uso nrun=0, NRUNTOT=1 per test/profiling.")
#     nrun = 0
#     NRUNTOT = 1





# === Scrive glasmaparameters.py solo per il primo run ===
if nrun == 0:
    with open("glasmaparameters.py", "w") as f:
        f.write(f"NRUNTOT = {NRUNTOT}\n")
        f.write(f"NSTEPS = {n_steps}\n")
        f.write(f"t0 = {t0}\n")

# === Crea directory per il run corrente ===
os.makedirs(f"evolution_run/evolution_run{nrun}", exist_ok=True)
# === Crea directory per le configurazioni HQ ===
os.makedirs("HQinitializations", exist_ok=True)


# === INIZIALIZZAZIONE ===
Utot_x, Utot_y, Utot_eta, Ex, Ey, Eeta = initialization()

# # Salviamo Utot_x, Utot_y, Eeta su file di testo in cartella apposita nel main
# save_matrix_configuration(Utot_x, f"Utot_x_{nrun}.txt", os.path.join("inizializzazioni_campi_gluoni",f"inizializzazione{nrun}"),"inizializzazione gauge links Utot_x")

# save_matrix_configuration(Utot_y, f"Utot_y_{nrun}.txt", os.path.join("inizializzazioni_campi_gluoni",f"inizializzazione{nrun}"),"inizializzazione gauge links Utot_y")

# save_matrix_configuration(Eeta, f"Eeta_{nrun}.txt", os.path.join("inizializzazioni_campi_gluoni",f"inizializzazione{nrun}"),"inizializzazione campo elettrico Eeta")


if cfg.HQUARKS.lower() == "yes":
    XHQ, YHQ, etaHQ, PXHQ, PYHQ, PetaHQ, Qmat = initialize_HQ_run(nrun)
    
    # q2 = np.real(np.einsum('aij,aji->a', Qmat, Qmat))
    # q2 *= 2.0
    
    
    """Momenta are written in GeV in the data files"""
    
    data = np.column_stack((XHQ, YHQ, hbarc * PXHQ, hbarc * PYHQ))
    fileHQ = f"HQinitializations/XandP_tinit_{nrun}.csv"
    np.savetxt(fileHQ, data, delimiter=",")
    
    print("----------------------------------")
    print(f"INIIALIZZAZIONE HQs EVENTO {nrun}")
    # print(f"q2: {np.mean(q2):.5f}, deviazione: {np.std(q2):.5e}")
    print("----------------------------------")
    
    
    
    
    


Ux_3d    = extend_in_eta(Utot_x, N_eta)
Uy_3d    = extend_in_eta(Utot_y, N_eta)
Ueta_3d  = extend_in_eta(Utot_eta, N_eta)


#===========================================================================================================================
# CODICE CON ATTACCO DELLE FLUTTUAZIONI ALL'ISTANTE INIZIALE, CIOE' NELL'INIZIALIZZAZIONE    (1)
#===========================================================================================================================

if i_fluct == 0:                           #no fluctuations
    Ex_3d    = extend_in_eta(Ex, N_eta)
    Ey_3d    = extend_in_eta(Ey, N_eta)
    Eeta_3d  = extend_in_eta(Eeta, N_eta)
else:                                      #with glasma fluctuations
    delta_Ex, delta_Ey, delta_Eeta = delta_fluctuations_glasma(Nx, Ny, N_eta, a_perp, a_eta, Delta, i_fluct, Ux_3d, Uy_3d, i_variance, lambdas)
    Ex_3d    = extend_in_eta(Ex, N_eta) + delta_Ex
    Ey_3d    = extend_in_eta(Ey, N_eta) + delta_Ey
    Eeta_3d  = extend_in_eta(Eeta, N_eta) + delta_Eeta

# #stima della variazione dei campi elettrici nel primo step evolutivo, senza usare leapfrog
# RHS_x, RHS_y, RHS_eta = compute_RHS_YM_at_tau0(Ux_3d, Uy_3d, Ueta_3d, g, a_perp, a_eta, t0, Nx, Ny, N_eta)
# Ex_t0piudt = Ex_3d + RHS_x * dt
# Ey_t0piudt = Ey_3d + RHS_y * dt
# Eeta_t0piudt = Eeta_3d + RHS_eta * dt
# #eps_t0 = calculate_spatial_observables(Ux_3d, Uy_3d, Ueta_3d, Ex_3d, Ey_3d, Eeta_3d, g, a_perp, a_eta, t0)[0]
# #eps_t0piudt = calculate_spatial_observables(Ux_3d, Uy_3d, Ueta_3d, Ex_t0piudt, Ey_t0piudt, Eeta_t0piudt, g, a_perp, a_eta, t0+dt)[0]
# trEx_t0 = calculate_trace_field(Ex_3d)
# trEy_t0 = calculate_trace_field(Ey_3d)
# trEx_t0piudt = calculate_trace_field(Ex_t0piudt)
# trEy_t0piudt = calculate_trace_field(Ey_t0piudt)
# ET2_t0 = ((trEx_t0 + trEy_t0)/(t0**2))*(hbarc**4)
# ET2_t0piudt = ((trEx_t0piudt + trEy_t0piudt)/((t0+dt)**2))*(hbarc**4)
# print(f"Num. evento: {nrun} ---> ET^2 (t0) = {ET2_t0} Gev^4 ; ET^2 (t0+dt) = {ET2_t0piudt} Gev^4")
# difference = ET2_t0piudt - ET2_t0
# print(f"Variazione: {difference} Gev^4")



#===========================================================================================================================
# CODICE CON ATTACCO DELLE FLUTTUAZIONI A UN ISTANTE DIVERSO DA QUELLO INIZIALE             (2)
#===========================================================================================================================

# Ex_3d    = extend_in_eta(Ex, N_eta)                          #unica inizializzazione, con o senza fluttuazioni. 
# Ey_3d    = extend_in_eta(Ey, N_eta)                          #inserisco le fluttuazioni a tau_fluct_start in evolve_glasma
# Eeta_3d  = extend_in_eta(Eeta, N_eta)

# #calcolo le fluttuazioni e lo step temporale in cui attaccarle, se necessario

# step_fluct = None
# delta_Ex = None
# delta_Ey = None
# delta_Eeta = None


# if i_fluct in (1, 2, 3):
#     delta_Ex, delta_Ey, delta_Eeta = delta_fluctuations_glasma(Nx, Ny, N_eta, a_perp, a_eta, Delta, i_fluct, Ux_3d, Uy_3d, i_variance)
#     step_fluct = int(round((tau_fluct_start - tau_start) / dt))
#     print(f"Le fluttuazioni verranno aggiunte allo step {step_fluct}, tau = {t0 + step_fluct * dt:.4f} fm")




#===========================================================================================================================
# INFO PER SETTING evolve_glasma CON (1) O (2) PER ATTACCO FLUTTUAZIONI

#(1): nella firma di evolve_glasma non mi serve nulla che non sia nel codice originale

#(2): nella firma di evolve_glasma servono step_fluct, Delta, i_fluct ---> per dare input a delta_fluctuations_glasma e a 
#     calculate_F_eta_model (che lavora dentro l'appena citata funzione)

# ricorda di commentare e decommentare le apposite righe nelle firme di evolve_glasma qui sotto e anche nella definizione
# oltre che per le firme, anche per la function in sè, quando scegli di usare la (1) o la (2)

#===========================================================================================================================





# === EVOLUZIONE CYM ===
pathwrite = f"evolution_run/evolution_run{nrun}/observables_vs_tau.csv"
csv_file = open(pathwrite, "w", newline="")
csv_writer = csv.writer(csv_file)

if cfg.HQUARKS.strip().lower() == "yes":
    Ux_f, Uy_f, Ueta_f, Ex_f, Ey_f, Eeta_f, XHQ_f, YHQ_f, \
        PXHQ_f, PYHQ_f, PetaHQ_f, QHQ_f = \
            evolve_glasma(
        Ux_3d, Uy_3d, Ueta_3d,
        Ex_3d, Ey_3d, Eeta_3d,
        g, a_perp, a_eta,
        t0, dt, n_steps, nrun,
        Nx, Ny, N_eta,
        compute_observable=True,
        XHQ=XHQ, YHQ=YHQ, etaHQ=etaHQ, PXHQ=PXHQ, PYHQ=PYHQ, PetaHQ=PetaHQ, QHQ=Qmat
    )
else:
    Ux_f, Uy_f, Ueta_f, Ex_f, Ey_f, Eeta_f = evolve_glasma(
        Ux_3d, Uy_3d, Ueta_3d,
        Ex_3d, Ey_3d, Eeta_3d,
        g, a_perp, a_eta,
        t0, dt, n_steps, nrun,
        Nx, Ny, N_eta,
        compute_observable=True
    )


csv_file.close()

