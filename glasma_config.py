#------------------------------------------------------------------------------
#
# GLASMA EVOLUTION -- FULL HOTSPOT MODEL
# parameter file
# file dei parametri
#------------------------------------------------------------------------------
#
# The global parameters are defined here.
# Qui sono definiti i parametri globali.
#
#------------------------------------------------------------------------------


# =============================================================================
# EMAIL controller
# =============================================================================


email_send = "no"

# =============================================================================
# GLOBAL PARAMETERS
# PARAMETRI GLOBALI
# =============================================================================

# --- Time evolution
# --- Evoluzione temporale
tau_start = 0.02        # Initial simulation time   [fm/c]
tf = 0.4                # Final simulation time in fm/c
n_steps = 400           # Without fluctuations use 200, with fluctuations 800
                        # senza fluct metti 200, con 800


# --- Frequency for saving observables
# --- Frequenza salvataggio osservabili
step_interval = 2       # Interval between output-writing steps in the CYM evolution   [Number of steps]
                        # misura l'intervallo per gli step di scrittura in CYM evolution
step_target = 1         # Time step at which 3D quantities are computed   [Step index]
                        # step temporale per calcolare quantità 3D

# --- Total number of events (to be set for parallel runs)
# --- Numero eventi totali (da settare in parallelo)
NRUNTOT = 1



# +----------------+----------+--------+
# | Centrality (%) | ⟨N_part⟩ |   σ    |
# | Centralità (%) |          |        |
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

# --- Collision parameters
# --- Parametri collisione
Npart_p = 1             # Number of participants in projectile p (for a proton Npart_p=1)
                        # numero di partecipanti nel proiettile p (per protone Npart_p=1)
Npart_A = 8


Nsheets = 5 #50            # NUMBER of SHEETS

# --- Transverse lattice parameters
# --- Parametri del reticolo trasverso
Lx = 3.0                # Physical dimensions of the transverse box   [fm]
                        # Dimensioni fisiche in fm

Nx = 64                 # Number of points in the transverse plane
                        # Numero di punti nel piano trasverso
Ly = Lx
Ny = Nx


# --- Parameters for the rapidity axis η
# --- Parametri per l'asse di rapidità η
Leta = 1
N_eta = 1               # Number of points in η (1 for boost invariance)
                        # Numero di punti in η (1 per boost-invarianza)




mIR = 0.2               # IR regulator mass   [GeV]


#------------------------------------------------------------------------------
# Heavy-quark parameters
# Parametri Heavy Quarks
#------------------------------------------------------------------------------
HQUARKS = "yes"

NHQ = 10 #20000             # Number of quarks plus number of antiquarks

plot_hq = "yes"

sigma_corr = 0.1        # Width of the antiquark coordinate-space distribution   [fm]
                        # width della distribuzione in coordinate degli antiquark (fm)

init_mom_space = "FONLL"

hq_flavor = "charm"

if hq_flavor == "charm":
    mass_HQ = 1.3       # Mass of HQ   [GeV]
if hq_flavor == "beauty":
    mass_HQ = 4.5       # Mass of HQ   [GeV]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#    MISCELLANEOUS SWITCHES
#    CONTROLLORI VARI
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

i_plot = 0              # i_plot=1 plots Tp
                        # i_plot=1 fa il plot della Tp
i_plot_finale = 0



i_filter = 0            # i_filter=0: no UV filter on rhok
                        # i_filter=1: Gaussian filter
                        # i_filter=2: massive-propagator filter
                        # i_filter=3: 1/k^2 filter
                        #
                        # i_filter=0: no filtro UV su rhok.
                        # i_filter=1: filtro gaussiano
                        # i_filter=2: filtro massive propagator
                        # i_filter=3: filtro 1/k^2

i_homogen_A = 1         # i_homogen=0: mu in projectile 2 follows Tp
                        # i_homogen=1: mu in projectile 2 is homogeneous in the transverse plane
                        # i_homogen=2: Gaussian tube (spherical proton)
                        #
                        # i_homogen=0: mu nel proiettile 2 segue la Tp
                        # i_homogen=1: mu nel proiettile 2 è omogeneo nel piano trasverso
                        # i_homogen=2: tubo gaussiano (spherical proton)
mu_A = 0.85             # Value of μ for the homogeneous configuration   [GeV]
                        # Valore di μ in GeV per configurazione omogenea


i_homogen_p = 0         # Same as i_homogen_A, for projectile 1
                        # same as i_homogen_A, per il proiettile 1


ITER_OR = 1             #------------------------------------------------------------------------------
                        # ITER_OR = 0: NO discrete transverse projection for the spectrum
                        # ITER_OR = 1: discrete transverse projection for the spectrum
                        #
                        # ITER_OR = 0: NO proiezione trasversa discreta per spettro
                        # ITER_OR = 1: proiezione trasversa discreta per spettro
                        #------------------------------------------------------------------------------



i_cho_spe_anal = 1


i_fluct = 0             # Selects the model for the strength of the glasma fluctuations
                        # sceglie il modello per la strength delle fluttuazioni sul glasma
                        #-------------------------------------------------------------------
                        # i_fluct = 0: NO fluctuations
                        # i_fluct = 1: model with 1 cosine
                        # i_fluct = 2: model with a superposition of 3 cosines
                        # i_fluct = 3: Gaussian random number
                        # i_fluct = 4: exponential correlation length in rapidity
                        #
                        # i_fluct = 1: modello con 1 coseno
                        # i_fluct = 2: modello con sovrapposizione di 3 coseni


# ------------------------------------------------------------------------------
#  GLASMA FLUCTUATION parameters
#  Parametri FLUTTUAZIONI SUL GLASMA
# ------------------------------------------------------------------------------

Delta = 0.05            # Fluctuation strength   [Dimensionless]
                        # strength della fluttuazione

lambdas = 0.5           # Correlation length in rapidity (i_fluct==4)   [Dimensionless]

tau_fluct_start = 0.02  # Time at which the fluctuation is switched on (different from tau_start of the leading-order glasma)   [fm]
                        # tempo al quale attaccare la fluttuazione (diverso da tau_start del glasma leading order)

i_variance = 1          # Selects the model used to generate the fluctuations in the transverse plane
                        # sceglie il modello per generare le fluttuazioni sul piano trasverso
                        #---------------------------------------------------------------------
                        # i_variance = 0: random number-> weight w=1 (Parisi)
                        # i_variance = 1: weight with Tp product
