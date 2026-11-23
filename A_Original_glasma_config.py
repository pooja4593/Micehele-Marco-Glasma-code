#------------------------------------------------------------------------------
#
# GLASMA EVOLUTION -- FULL HOTSPOT MODEL
# file dei parametri
#------------------------------------------------------------------------------
#
# Qui sono definiti i parametri globali. 
# 
#------------------------------------------------------------------------------


# =============================================================================
# EMAIL controller
# =============================================================================


email_send = "no"

# =============================================================================
# PARAMETRI GLOBALI
# =============================================================================

# --- Evoluzione temporale
tau_start = 0.02
tf = 0.4
n_steps = 400  #senza fluct metti 200, con 800


# --- Frequenza salvataggio osservabili
step_interval = 2   # misura l'intervallo per gli step di scrittura in CYM evolution
step_target = 1      # step temporale per calcolare quantità 3D

# --- Numero eventi totali (da settare in parallelo)
NRUNTOT = 1



# +----------------+----------+--------+
# | Centralità (%) | ⟨N_part⟩ |   σ    |
# +----------------+----------+--------+
# | 0–5            |   15.7   |  3.84  |
# | 5–10           |   14.0   |  3.78  |
# | 10–20          |   12.7   |  3.85  |
# | 20–40          |   10.4   |  3.93  |
# | 40–60          |   7.42   |  3.61  |
# | 60–80          |   4.81   |  2.69  |
# | 80–100         |   2.94   |  1.42  |
# | 0–100          |   7.87   |  5.10  |
# +----------------+----------+--------+

# --- Parametri collisione
Npart_p = 1       # numero di partecipanti nel proiettile p (per protone Npart_p=1)
Npart_A = 8


Nsheets = 50      # NUMBER of SHEETS

# --- Parametri del reticolo trasverso
Lx = 3.0
Ly = Lx                    # Dimensioni fisiche in fm
Nx = 64                    # Numero di punti nel piano trasverso
Ny = Nx
 

# --- Parametri per l'asse di rapidità η
Leta = 1
N_eta = 1                  # Numero di punti in η (1 per boost-invarianza)




mIR = 0.2        # GeV


#------------------------------------------------------------------------------
# Parametri Heavy Quarks
HQUARKS = "yes"                    
NHQ = 20000    # number of quarks plus number of antiquarks

plot_hq = "yes"

sigma_corr = 0.1  # width della distribuzione in coordinate degli antiquark (fm)

init_mom_space = "FONLL"

hq_flavor = "charm"

if hq_flavor == "charm":
    mass_HQ = 1.3 # GeV
if hq_flavor == "beauty":
    mass_HQ = 4.5 # GeV
#------------------------------------------------------------------------------

# -------------------------------------------------------
#    CONTROLLORI VARI
# -------------------------------------------------------
i_plot = 0           # i_plot=1 fa il plot della Tp
i_plot_finale = 0

 

i_filter = 0         # i_filter=0: no filtro UV su rhok. 
                     # i_filter=1: filtro gaussiano
                     # i_filter=2: filtro massive propagator
                     # i_filter=3: filtro 1/k^2

i_homogen_A = 1      # i_homogen=0: mu nel proiettile 2 segue la Tp
                     # i_homogen=1: mu nel proiettile 2 è omogeneo nel piano trasverso
                     # i_homogen=2: tubo gaussiano (spherical proton)
mu_A = 0.85           # Valore di μ in GeV per configurazione omogenea


i_homogen_p = 0     # same as i_homogen_A, per il proiettile 1


ITER_OR = 1         #------------------------------------------------------------------------------
                    # ITER_OR = 0: NO proiezione trasversa discreta per spettro
                    # ITER_OR = 1: proiezione trasversa discreta per spettro
                    #------------------------------------------------------------------------------
                    

                    
i_cho_spe_anal = 1


i_fluct = 0         #sceglie il modello per la strength delle fluttuazioni sul glasma
                    #-------------------------------------------------------------------
                    # i_fluct = 0: NO fluctuations
                    # i_fluct = 1: modello con 1 coseno
                    # i_fluct = 2: modello con sovrapposizione di 3 coseni
                    # i_fluct = 3: gaussian random number
                    # i_fluct = 4: exponential correlation length in rapidity


# ------------------------------------------------------------------------------
#  Parametri FLUTTUAZIONI SUL GLASMA
# ------------------------------------------------------------------------------

Delta = 0.05             #strength della fluttuazione

lambdas = 0.5            #correlation length (i_fluct==4)

tau_fluct_start = 0.02   #tempo al quale attaccare la fluttuazione (diverso da tau_start del glasma leading order)

i_variance = 1      # sceglie il modello per generare le fluttuazioni sul piano trasverso
                    #---------------------------------------------------------------------
                    # i_variance = 0: random number-> weight w=1 (Parisi)
                    # i_variance = 1: weight with Tp product 
