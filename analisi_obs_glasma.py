#------------------------------------------------------------------------------
#
# GLASMA EVOLUTION -- FULL HOTSPOT MODEL
# analisi e plots
#------------------------------------------------------------------------------
#
# Questa routine fa l'analisi produce i plots necessari.  
# In particolare, calcola media e deviazione standard di quantità globali
# (pressioni, densità di energia, eccentricità,...).
# 
#------------------------------------------------------------------------------



import sys
import matplotlib
matplotlib.use('Agg')  # forza backend non-GUI (salva su file invece di mostrare a schermo)

import os
import pandas as pd
import glasmaparameters as par  # questo deve contenere NRUNTOT
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import rcParams

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",  # o 'stixsans', 'cm', ecc.
    "axes.labelsize": 30,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "lines.linewidth": 2.5
})

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # TIMESTAMP comune 

#------------------------------------------------------------------------------
# MEDIE
#------------------------------------------------------------------------------

dataframes = []


folderave="evolutionaverage"   # in questa cartella salva i file mediati
fileaverage = os.path.join(folderave, "evolution_average.csv")  # qui scrive le medie

os.makedirs(folderave, exist_ok=True)   # Non fa nulla se la directory già esiste

# === Determina NSTEPS dal primo file (run 0) ===
first_path = "evolution_run/evolution_run0/observables_vs_tau.csv"
if not os.path.exists(first_path):
    raise FileNotFoundError("Run 0 file does not exist / Il file del run 0 non esiste: evolution_run/evolution_run0/observables_vs_tau.csv")

first_df = pd.read_csv(first_path, header=None)
NSTEPS, NOBS = first_df.shape
print(f"[INFO] Reference file: {NSTEPS} time steps, {NOBS} observables / File di riferimento: {NSTEPS} step temporali, {NOBS} osservabili.")

# Aggiungilo subito ai dataframe
dataframes = [first_df]

for nrun in range(1, par.NRUNTOT):
    path = f"evolution_run/evolution_run{nrun}/observables_vs_tau.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, header=None)
        dataframes.append(df)


for nrun, df in enumerate(dataframes):
    print(f"Run {nrun}: shape = {df.shape}")

# sys.exit()


if dataframes:
    mean_df = sum(dataframes) / len(dataframes)
    mean_df.to_csv(fileaverage, index=False, header=False)  # qui scrive le medie
    print("Mean saved to / Media salvata in evolution_average.csv")
    
    # === Calcolo e salvataggio della deviazione standard ===
    all_data = np.stack([df.values for df in dataframes])  # shape: (NRUN, Nstep, Nobs)
    # std_array = np.std(all_data, axis=0)  # shape: (Nstep, Nobs)
    Nev = len(dataframes)
    std_array = np.std(all_data, axis=0, ddof=1) / np.sqrt(Nev)


    std_df = pd.DataFrame(std_array)
    std_df.to_csv("evolutionaverage/evolution_std.csv", index=False, header=False)
    print("Standard deviation saved to / Deviazione standard salvata in evolution_std.csv")
    
else:
    print("No files found / Nessun file trovato.")
    
    
#------------------------------------------------------------------------------
# GRAFICI
#------------------------------------------------------------------------------   
    
data = np.loadtxt(fileaverage, delimiter=",")
tau, tr_ex, tr_ey, tr_eta, ener_dens_ele, blsq, btsq, ener_dens_tot, PT, PL, eps2, eps3, eps2T = data.T
std_data = np.loadtxt("evolutionaverage/evolution_std.csv", delimiter=",")
eps_std  = std_data[:, 7]
eps_pt = std_data[:, 8]
eps_pl = std_data[:, 9]
eps2_std = std_data[:, 10]
eps3_std = std_data[:, 11]
eps2T_std = std_data[:, 12]

eps_low  = ener_dens_tot - eps_std
eps_high = ener_dens_tot + eps_std
pl_low = PL - eps_pl
pl_high = PL + eps_pl
pt_low = PT - eps_pt
pt_high = PT + eps_pt
eps2_low = eps2 - eps2_std
eps2_high = eps2 + eps2_std
eps3_low = eps3 - eps3_std
eps3_high = eps3 + eps3_std
eps2T_low = eps2T - eps2T_std
eps2T_high = eps2T + eps2T_std


lw=3.0
lw_red=0.95 * lw

ls_dashed=(0, (6, 3))
ls_ds=(0, (6, 2, 1, 2))

#------------------------------------------------------------------------------
# eps2T with std dev
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)
ax.fill_between(tau, eps2T_low, eps2T_high, alpha=0.3, color="#5F9EA0")
ax.plot(tau, eps2T, label="$\\varepsilon_2^T$", linewidth=lw)


plt.xlabel("$\\tau$ [fm]")
# plt.ylabel("$\\varepsilon_i$ ")

plt.tight_layout()

plt.legend(
    frameon=True,
    facecolor='white',   # colore interno (es. 'white', 'gainsboro', 'none')
    edgecolor='gray',         # bordo
    framealpha=0.7,           # trasparenza: 1 = opaco, 0 = invisibile
    fancybox=True             # angoli arrotondati
)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

textstr = '\n'.join((
    r'$N_x \times N_y = 32 \times 32$',
    r'$N_\mathrm{sheets} = 50$',r'$N_p = 1 x 4$'
))
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
plt.text(
    0.73, 0.8, textstr,
    transform=plt.gca().transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=props
)

ax.text(
    0.5, 0.9, "pA collision", transform=ax.transAxes,
    fontsize=34, style='italic', ha='center',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.85)
)


filename = "evolutionaverage/ave_eps2t_16x16.png"
filenamepdf = "evolutionaverage/ave_eps2t_16x16.pdf"

# Salvataggio in PNG ad alta risoluzione
plt.savefig(filename, dpi=300)
plt.savefig(filenamepdf, bbox_inches='tight')
print(f"Plot saved to / Grafico salvato in: {filename}")


# plt.show()


# sys.exit()   

#------------------------------------------------------------------------------
# eps2 with std dev
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)
ax.fill_between(tau, eps2_low, eps2_high, alpha=0.3, color="#5F9EA0")
ax.plot(tau, eps2, label="$\\varepsilon_2$", linewidth=lw)
ax.fill_between(tau, eps3_low, eps3_high, alpha=0.3, color="#D9A66C")
ax.plot(tau, eps3, label="$\\varepsilon_3$", linewidth=lw)

plt.xlabel("$\\tau$ [fm]")
# plt.ylabel("$\\varepsilon_i$ ")

plt.tight_layout()

plt.legend(
    frameon=True,
    facecolor='white',   # colore interno (es. 'white', 'gainsboro', 'none')
    edgecolor='gray',         # bordo
    framealpha=0.7,           # trasparenza: 1 = opaco, 0 = invisibile
    fancybox=True             # angoli arrotondati
)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

textstr = '\n'.join((
    r'$N_x \times N_y = 32 \times 32$',
    r'$N_\mathrm{sheets} = 50$',r'$N_p = 1 x 4$'
))
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
plt.text(
    0.73, 0.8, textstr,
    transform=plt.gca().transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=props
)

ax.text(
    0.5, 0.9, "pA collision", transform=ax.transAxes,
    fontsize=34, style='italic', ha='center',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.85)
)


filename = "evolutionaverage/ave_eps2_16x16.png"
filenamepdf = "evolutionaverage/ave_eps2_16x16.pdf"

# Salvataggio in PNG ad alta risoluzione
plt.savefig(filename, dpi=300)
plt.savefig(filenamepdf, bbox_inches='tight')
print(f"Plot saved to / Grafico salvato in: {filename}")




# plt.show()

# sys.exit()



# PRESSURES and ENERGY DENSITY with std dev


# plt.figure(figsize=(10,8))
fig, ax = plt.subplots(figsize=(10, 8))

ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)
ax.fill_between(tau, eps_low, eps_high, alpha=0.3, color="#5F9EA0")
ax.plot(tau, ener_dens_tot, label="$\\varepsilon$", linewidth=lw)
ax.fill_between(tau, pt_low, pt_high, alpha=0.3, color="#f4a582")
ax.plot(tau, PT, label="$P_T$ ",linestyle=ls_dashed, linewidth=lw)
ax.fill_between(tau, pl_low, pl_high, alpha=0.3, color="#2E8B8B")
ax.plot(tau, PL, label="$P_L$", linestyle=ls_ds, linewidth=lw)
plt.xlabel("$\\tau$ [fm/c]")
plt.ylabel("[GeV⁴]")

plt.tight_layout()

plt.legend(
    frameon=True,
    facecolor='white',   # colore interno (es. 'white', 'gainsboro', 'none')
    edgecolor='gray',         # bordo
    framealpha=0.7,           # trasparenza: 1 = opaco, 0 = invisibile
    fancybox=True             # angoli arrotondati
)




plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

textstr = '\n'.join((
    r'$N_x \times N_y = 64 \times 64$',
    r'$N_\mathrm{sheets} = 50$'
))
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
plt.text(
    0.7, 0.2, textstr,
    transform=plt.gca().transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=props
)

ax.text(
    0.5, 0.9, "AA collision", transform=ax.transAxes,
    fontsize=34, style='italic', ha='center',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.85)
)




filename = "evolutionaverage/ave_glasma_press.png"
filenamepdf = "evolutionaverage/ave_glasma_press.pdf"

# Salvataggio in PNG ad alta risoluzione
plt.savefig(filename, dpi=300)
plt.savefig(filenamepdf, bbox_inches='tight')
print(f"Plot saved to / Grafico salvato in: {filename}")




# plt.show()

# sys.exit()

# PL/PT


# plt.figure(figsize=(10,8))
fig, ax = plt.subplots(figsize=(10, 8))

ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)
ax.axhline(y=0.5, color='black', linestyle=':', linewidth=1.0)
ax.plot(tau, PL/PT, label="$P_L/P_T$ ",linestyle='-', linewidth=lw)
ax.plot(tau, PL/ener_dens_tot, label="$P_L/\\varepsilon$ ",linestyle=ls_ds, linewidth=lw_red)
ax.plot(tau, PT/ener_dens_tot, label="$P_T/\\varepsilon$ ",linestyle=ls_dashed, linewidth=lw_red)
plt.xlabel("$\\tau$ [fm/c]")
# plt.ylabel("[GeV⁴]")

plt.tight_layout()

plt.legend(
    frameon=True,
    facecolor='white',   # colore interno (es. 'white', 'gainsboro', 'none')
    edgecolor='gray',         # bordo
    framealpha=0.7,           # trasparenza: 1 = opaco, 0 = invisibile
    fancybox=True             # angoli arrotondati
)




plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

textstr = '\n'.join((
    r'$N_x \times N_y = 32 \times 32$',
    r'$N_\mathrm{sheets} = 50$',
    r'$N_p = 8$'
))
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
plt.text(
    0.7, 0.2, textstr,
    transform=plt.gca().transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=props
)

ax.text(
    0.5, 0.9, "pA collision", transform=ax.transAxes,
    fontsize=34, style='italic', ha='center',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.85)
)




filename = "evolutionaverage/ave_glasma_plsupt.png"
filenamepdf = "evolutionaverage/ave_glasma_plsupt.pdf"

# Salvataggio in PNG ad alta risoluzione
plt.savefig(filename, dpi=300)
plt.savefig(filenamepdf, bbox_inches='tight')

print(f"Plot saved to / Grafico salvato in: {filename}")




# plt.show()

# sys.exit()


# FIELDS no inset

# plt.figure(figsize=(10,8))
fig, ax = plt.subplots(figsize=(10, 8))

plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)

plt.plot(tau, tr_eta, label="$E_L^2$")
plt.plot(tau, blsq, label="$B_L^2$")
plt.plot(tau, tr_ex + tr_ey, label="$E_T^2$", linestyle=ls_dashed)
plt.plot(tau, btsq, label="$B_T^2$", linestyle=ls_ds)
plt.xlabel("$\\tau$ [fm/c]")
plt.ylabel("[GeV⁴]")

plt.tight_layout()

plt.legend(
    frameon=True,
    facecolor='white',   # colore interno (es. 'white', 'gainsboro', 'none')
    edgecolor='gray',         # bordo
    framealpha=0.7,           # trasparenza: 1 = opaco, 0 = invisibile
    fancybox=True             # angoli arrotondati
)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

textstr = '\n'.join((
    r'$N_x \times N_y = 64 \times 64$',
    r'$N_\mathrm{sheets} = 50$',r'$1-8$'
))
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
plt.text(
    0.7, 0.5, textstr,
    transform=plt.gca().transAxes,
    fontsize=18,
    verticalalignment='top',
    bbox=props
)

ax.text(
    0.5, 0.9, "pA collision", transform=ax.transAxes,
    fontsize=34, style='italic', ha='center',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.85)
)


filename = "evolutionaverage/ave_glasma_campi.png"
filenamepdf = "evolutionaverage/ave_glasma_campi.pdf"

# Salvataggio in PNG ad alta risoluzione
plt.savefig(filename, dpi=300)
plt.savefig(filenamepdf, bbox_inches='tight')
print(f"Plot saved to / Grafico salvato in: {filename}")


# plt.show()






