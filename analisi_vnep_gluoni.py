# ───────────────────────────────────────────────────────────────
# 1) Parametri e binning (come prima)
# ───────────────────────────────────────────────────────────────
import numpy as np, pandas as pd, glob, os
import matplotlib
matplotlib.use('Agg')  # usa backend non interattivo

import matplotlib.pyplot as plt

import glasma_config as cfg

number_events = cfg.NRUNTOT
n_spec_used = cfg.i_cho_spe_anal

Lx, Nx = cfg.Lx, cfg.Nx
a_fm       = Lx / Nx                  # [fm]
fm_to_GeV  = 5.067730937              # 1 fm = 5.06773 GeV⁻¹
a          = a_fm * fm_to_GeV         # lattice spacing [GeV⁻¹]

delta_kT   = 0.5                      # larghezza del bin (GeV, lattice)
files      = sorted(glob.glob('spectrum_event/spectrum_event[0-9]*.csv'))

# cartella di output per i file del vn dei gluoni
out_dir = "vn_gluoni"
os.makedirs(out_dir, exist_ok=True)

# Stima del massimo kT presente nei dati
df_tmp = pd.read_csv(
    files[0],
    names=['kx','ky','dNd2k_FULL', 'dNd2k_noODD', 'dNd2k_EQUIPA'],
    sep=',',
    skipinitialspace=True
    ).astype(float)

kT_max_data = np.sqrt(df_tmp.kx.values**2 + df_tmp.ky.values**2).max()
kT_max = np.ceil(kT_max_data / delta_kT) * delta_kT

# Definizione dei bin
IR_lim = 0.25 # IR cutoff 
kT_bins    = np.arange(IR_lim, kT_max + delta_kT, delta_kT)
kT_centres = 0.5 * (kT_bins[1:] + kT_bins[:-1])
# ───────────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────────
# 2) Nuova funzione per calcolare v2 con EP esplicito
# ───────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────
# 2.1) Funzione ottimizzata per Psi_EP
# ───────────────────────────────────────────────────────────────
def calculate_Psi_EP_n(kx, ky, weights, n, kT_min, kT_max):
    """Calcola l'angolo del piano di evento con soppressione del rumore.
    Args:
        kx, ky: array delle componenti del momento trasverso.
        weights: array dei pesi (dNd2k).
        n: ordine dell'armonica
        kT_min, kT_max: range di kT usato per calcolare Psi_n
    Returns:
        psi_ep: angolo in radianti, in [0, π).
                                        
    """
    
    kT = np.sqrt(kx**2 + ky**2)
    filter_EP = (kT >= kT_min) & (kT <= kT_max)
     
    # kx = kx + reg_IR
    phi = np.arctan2(ky[filter_EP], kx[filter_EP])
    
    Qx = np.sum(weights[filter_EP] *   np.cos (n * phi)   ) 
    Qy = np.sum(weights[filter_EP] * np.sin (n * phi) )
    psi_ep = (1.0 / n) * np.arctan2(Qy, Qx)  # Attenzione a usare arctan2(y,x)
    # psi_ep = psi_ep % np.pi
    return psi_ep

# ───────────────────────────────────────────────────────────────
# 2.2) Funzione principale per vn con EP
# ───────────────────────────────────────────────────────────────
def calculate_vn_EP_method(df, kT_bins, n, ntype, kT_min=1.0, kT_max=6.0):
    """Calcola v2(kT) con il metodo EP, usando calculate_Psi_EP."""
    kx = df.kx.values
    ky = df.ky.values
    
    if ntype == 1:
        dNd2k = df.dNd2k_FULL.values
    elif ntype == 2:
        dNd2k = df.dNd2k_noODD.values
    elif ntype == 3:
        dNd2k = df.dNd2k_EQUIPA.values
        
    kT = np.sqrt(kx**2 + ky**2)
    
    # Usa la nuova funzione per Psi_EP
    Psi_EP_2 = calculate_Psi_EP_n(kx, ky, dNd2k, n, kT_min, kT_max)  
    
    # Resto del codice invariato...
    vn_values = np.full(len(kT_bins)-1, np.nan)
    for i, (k_lo, k_hi) in enumerate(zip(kT_bins[:-1], kT_bins[1:])):
        mask = (kT >= k_lo) & (kT < k_hi)
        if np.sum(mask) > 0:
            phi = np.arctan2(ky[mask], kx[mask])
            vn_values[i] = np.average(np.cos(n*(phi - Psi_EP_2)), 
                                      weights=dNd2k[mask])
    return vn_values

#==============================================================================
# 3) Main: calcola vn con cut (1,6) GeV per Psi_n
#==============================================================================


v1_EP_evt = []
v2_EP_evt = []   
v3_EP_evt = []
v4_EP_evt = []
v5_EP_evt = []
v6_EP_evt = []
v7_EP_evt = []
for f in files:
    df = pd.read_csv(
        f,
        names=['kx','ky','dNd2k_FULL', 'dNd2k_noODD', 'dNd2k_EQUIPA'],
        sep=',',
        skipinitialspace=True
        ).astype(float)
    
    print(f"Processing {f}")
    
    # Calcola con entrambi i metodi
    v1_EP_i = calculate_vn_EP_method(df, kT_bins, 1, n_spec_used)
    v1_EP_evt.append(v1_EP_i)
    v2_EP_i = calculate_vn_EP_method(df, kT_bins, 2, n_spec_used)
    v2_EP_evt.append(v2_EP_i)
    v3_EP_i = calculate_vn_EP_method(df, kT_bins, 3, n_spec_used)
    v3_EP_evt.append(v3_EP_i)
    v4_EP_i = calculate_vn_EP_method(df, kT_bins, 4, n_spec_used)
    v4_EP_evt.append(v4_EP_i)
    v5_EP_i = calculate_vn_EP_method(df, kT_bins, 5, n_spec_used)
    v5_EP_evt.append(v5_EP_i)
    v6_EP_i = calculate_vn_EP_method(df, kT_bins, 6, n_spec_used)
    v6_EP_evt.append(v6_EP_i)
    v7_EP_i = calculate_vn_EP_method(df, kT_bins, 7, n_spec_used)
    v7_EP_evt.append(v7_EP_i)


#------------------------------------------------------------------------------
# scrittura file v1
#------------------------------------------------------------------------------
v1_EP_evt = np.array(v1_EP_evt)
# Medie sugli eventi
v1_EP_mean = np.nanmean(v1_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v1_EP_evt[:, 0]))
v1_EP_err = np.nanstd(v1_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v1_EP':        v1_EP_mean,
    'v1_EP_err':    v1_EP_err
})
out.to_csv(os.path.join(out_dir, f'v1_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v1_kT_{number_events}.csv")


#------------------------------------------------------------------------------
# scrittura file v2
#------------------------------------------------------------------------------
v2_EP_evt = np.array(v2_EP_evt)
# Medie sugli eventi
v2_EP_mean = np.nanmean(v2_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v2_EP_evt[:, 0]))
v2_EP_err = np.nanstd(v2_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v2_EP':        v2_EP_mean,
    'v2_EP_err':    v2_EP_err
})
out.to_csv(os.path.join(out_dir, f'v2_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v2_kT_{number_events}.csv")

#------------------------------------------------------------------------------
# scrittura file v3
#------------------------------------------------------------------------------
v3_EP_evt = np.array(v3_EP_evt)
# Medie sugli eventi
v3_EP_mean = np.nanmean(v3_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v3_EP_evt[:, 0]))
v3_EP_err = np.nanstd(v3_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v3_EP':        v3_EP_mean,
    'v3_EP_err':    v3_EP_err
})
out.to_csv(os.path.join(out_dir, f'v3_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v3_kT_{number_events}.csv")

#------------------------------------------------------------------------------
# scrittura file v4
#------------------------------------------------------------------------------
v4_EP_evt = np.array(v4_EP_evt)
# Medie sugli eventi
v4_EP_mean = np.nanmean(v4_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v4_EP_evt[:, 0]))
v4_EP_err = np.nanstd(v4_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v4_EP':        v4_EP_mean,
    'v4_EP_err':    v4_EP_err
})
out.to_csv(os.path.join(out_dir, f'v4_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v4_kT_{number_events}.csv")

#------------------------------------------------------------------------------
# scrittura file v5
#------------------------------------------------------------------------------
v5_EP_evt = np.array(v5_EP_evt)
# Medie sugli eventi
v5_EP_mean = np.nanmean(v5_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v5_EP_evt[:, 0]))
v5_EP_err = np.nanstd(v5_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v5_EP':        v5_EP_mean,
    'v5_EP_err':    v5_EP_err
})
out.to_csv(os.path.join(out_dir, f'v5_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v5_kT_{number_events}.csv")


#------------------------------------------------------------------------------
# scrittura file v6
#------------------------------------------------------------------------------
v6_EP_evt = np.array(v6_EP_evt)
# Medie sugli eventi
v6_EP_mean = np.nanmean(v6_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v6_EP_evt[:, 0]))
v6_EP_err = np.nanstd(v6_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v6_EP':        v6_EP_mean,
    'v6_EP_err':    v6_EP_err
})
out.to_csv(os.path.join(out_dir, f'v6_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v6_kT_{number_events}.csv")


#------------------------------------------------------------------------------
# scrittura file v7
#------------------------------------------------------------------------------
v7_EP_evt = np.array(v7_EP_evt)
# Medie sugli eventi
v7_EP_mean = np.nanmean(v7_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v7_EP_evt[:, 0]))
v7_EP_err = np.nanstd(v7_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v7_EP':        v7_EP_mean,
    'v7_EP_err':    v7_EP_err
})
out.to_csv(os.path.join(out_dir, f'v7_kT_{number_events}.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v7_kT_{number_events}.csv")


#==============================================================================

#==============================================================================
# Plot vn
#==============================================================================
plt.figure(figsize=(10, 6))
plt.axhline(0, color='brown', linestyle='--', linewidth=1.2, label=None)

plt.errorbar(kT_centres, v1_EP_mean, yerr=v1_EP_err, marker='o', label='v_1', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v2_EP_mean, yerr=v2_EP_err, marker='$\\heartsuit$', label='v_2', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v3_EP_mean, yerr=v3_EP_err, marker='^', label='v_3', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v4_EP_mean, yerr=v4_EP_err, marker='D', label='v_4', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v5_EP_mean, yerr=v5_EP_err, marker='p', 
             label='v_5', 
             linestyle='--', capsize=4, markersize=9)
plt.errorbar(kT_centres, v6_EP_mean, yerr=v6_EP_err, marker='h', label='v_6', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v7_EP_mean, yerr=v7_EP_err, marker='$\\clubsuit$', label='v_7', 
             linestyle='--', capsize=3)
plt.xlabel(r'$k_T\,[\mathrm{GeV}]$')
plt.ylabel(r'$v_n(k_T)$')

plt.xlim(0.0,8.0)
plt.ylim(-0.02,0.2)

plt.title(r'$\tau=0.4$ fm')
plt.grid(alpha=0.4, ls='--')
plt.legend()
plt.tight_layout()
filename_v2 = os.path.join(out_dir, f'vn_kT_{number_events}_nspec_{n_spec_used}.png')
filename_v2_pdf = os.path.join(out_dir, f'vn_kT_{number_events}_nspec_{n_spec_used}.pdf')
plt.savefig(filename_v2, dpi=300)
plt.savefig(filename_v2_pdf)
# plt.show()
#==============================================================================



#==============================================================================
# 3) Main: calcola vn SENZA cut per Psi_n
#==============================================================================
v1_EP_evt = []
v2_EP_evt = []   
v3_EP_evt = []
v4_EP_evt = []
v5_EP_evt = []
v6_EP_evt = []
v7_EP_evt = []
for f in files:
    df = pd.read_csv(
        f,
        names=['kx','ky','dNd2k_FULL', 'dNd2k_noODD', 'dNd2k_EQUIPA'],
        sep=',',
        skipinitialspace=True
        ).astype(float)
    
    print(f"Processing {f}")
    
    # Calcola con entrambi i metodi
    v1_EP_i = calculate_vn_EP_method(df, kT_bins, 1, n_spec_used, 0.01, 13.0)
    v1_EP_evt.append(v1_EP_i)
    v2_EP_i = calculate_vn_EP_method(df, kT_bins, 2, n_spec_used, 0.01, 13.0)
    v2_EP_evt.append(v2_EP_i)
    v3_EP_i = calculate_vn_EP_method(df, kT_bins, 3, n_spec_used, 0.01, 13.0)
    v3_EP_evt.append(v3_EP_i)
    v4_EP_i = calculate_vn_EP_method(df, kT_bins, 4, n_spec_used, 0.01, 13.0)
    v4_EP_evt.append(v4_EP_i)
    v5_EP_i = calculate_vn_EP_method(df, kT_bins, 5, n_spec_used, 0.01, 13.0)
    v5_EP_evt.append(v5_EP_i)
    v6_EP_i = calculate_vn_EP_method(df, kT_bins, 6, n_spec_used, 0.01, 13.0)
    v6_EP_evt.append(v6_EP_i)
    v7_EP_i = calculate_vn_EP_method(df, kT_bins, 7, n_spec_used, 0.01, 13.0)
    v7_EP_evt.append(v5_EP_i)


#------------------------------------------------------------------------------
# scrittura file v1
#------------------------------------------------------------------------------
v1_EP_evt = np.array(v1_EP_evt)
# Medie sugli eventi
v1_EP_mean = np.nanmean(v1_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v1_EP_evt[:, 0]))
v1_EP_err = np.nanstd(v1_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v1_EP':        v1_EP_mean,
    'v1_EP_err':    v1_EP_err
})
out.to_csv(os.path.join(out_dir, f'v1_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v1_kT_{number_events}_SENZACUT.csv")


#------------------------------------------------------------------------------
# scrittura file v2
#------------------------------------------------------------------------------
v2_EP_evt = np.array(v2_EP_evt)
# Medie sugli eventi
v2_EP_mean = np.nanmean(v2_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v2_EP_evt[:, 0]))
v2_EP_err = np.nanstd(v2_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v2_EP':        v2_EP_mean,
    'v2_EP_err':    v2_EP_err
})
out.to_csv(os.path.join(out_dir, f'v2_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v2_kT_{number_events}_SENZACUT.csv")

#------------------------------------------------------------------------------
# scrittura file v3
#------------------------------------------------------------------------------
v3_EP_evt = np.array(v3_EP_evt)
# Medie sugli eventi
v3_EP_mean = np.nanmean(v3_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v3_EP_evt[:, 0]))
v3_EP_err = np.nanstd(v3_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v3_EP':        v3_EP_mean,
    'v3_EP_err':    v3_EP_err
})
out.to_csv(os.path.join(out_dir, f'v3_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v3_kT_{number_events}_SENZACUT.csv")

#------------------------------------------------------------------------------
# scrittura file v4
#------------------------------------------------------------------------------
v4_EP_evt = np.array(v4_EP_evt)
# Medie sugli eventi
v4_EP_mean = np.nanmean(v4_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v4_EP_evt[:, 0]))
v4_EP_err = np.nanstd(v4_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v4_EP':        v4_EP_mean,
    'v4_EP_err':    v4_EP_err
})
out.to_csv(os.path.join(out_dir, f'v4_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v4_kT_{number_events}_SENZACUT.csv")

#------------------------------------------------------------------------------
# scrittura file v5
#------------------------------------------------------------------------------
v5_EP_evt = np.array(v5_EP_evt)
# Medie sugli eventi
v5_EP_mean = np.nanmean(v5_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v5_EP_evt[:, 0]))
v5_EP_err = np.nanstd(v5_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v5_EP':        v5_EP_mean,
    'v5_EP_err':    v5_EP_err
})
out.to_csv(os.path.join(out_dir, f'v5_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v5_kT_{number_events}_SENZACUT.csv")

#------------------------------------------------------------------------------
# scrittura file v6
#------------------------------------------------------------------------------
v6_EP_evt = np.array(v6_EP_evt)
# Medie sugli eventi
v6_EP_mean = np.nanmean(v6_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v6_EP_evt[:, 0]))
v6_EP_err = np.nanstd(v6_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v6_EP':        v6_EP_mean,
    'v6_EP_err':    v6_EP_err
})
out.to_csv(os.path.join(out_dir, f'v6_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v6_kT_{number_events}_SENZACUT.csv")

#------------------------------------------------------------------------------
# scrittura file v7
#------------------------------------------------------------------------------
v7_EP_evt = np.array(v7_EP_evt)
# Medie sugli eventi
v7_EP_mean = np.nanmean(v7_EP_evt, axis=0)
# Errori statistici
N_evt = np.sum(~np.isnan(v7_EP_evt[:, 0]))
v7_EP_err = np.nanstd(v7_EP_evt, axis=0, ddof=1) / np.sqrt(N_evt)
# Salva risultati
out = pd.DataFrame({
    'kT_latt[GeV]': kT_centres,
    'v7_EP':        v7_EP_mean,
    'v7_EP_err':    v7_EP_err
})
out.to_csv(os.path.join(out_dir, f'v7_kT_{number_events}_SENZACUT.csv'), index=False)
print(f"Results saved to / Risultati salvati in vn_gluoni/v7_kT_{number_events}_SENZACUT.csv")


#==============================================================================

#==============================================================================
# Plot vn
#==============================================================================
plt.figure(figsize=(10, 6))
plt.axhline(0, color='brown', linestyle='--', linewidth=1.2, label=None)

plt.errorbar(kT_centres, v1_EP_mean, yerr=v1_EP_err, marker='o', label='v_1', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v2_EP_mean, yerr=v2_EP_err, marker='$\\heartsuit$', label='v_2', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v3_EP_mean, yerr=v3_EP_err, marker='^', label='v_3', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v4_EP_mean, yerr=v4_EP_err, marker='D', label='v_4', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v5_EP_mean, yerr=v5_EP_err, marker='p', 
             label='v_5', 
             linestyle='--', capsize=4, markersize=9)
plt.errorbar(kT_centres, v6_EP_mean, yerr=v6_EP_err, marker='h', label='v_6', 
             linestyle='--', capsize=3)
plt.errorbar(kT_centres, v7_EP_mean, yerr=v7_EP_err, marker='$\\clubsuit$', label='v_7', 
             linestyle='--', capsize=3)
plt.xlabel(r'$k_T\,[\mathrm{GeV}]$')
plt.ylabel(r'$v_n(k_T)$')

plt.xlim(0.0,8.0)
plt.ylim(-0.02,0.2)

plt.title(r'$\tau=0.4$ fm')
plt.grid(alpha=0.4, ls='--')
plt.legend()
plt.tight_layout()
filename_v2 = os.path.join(out_dir, f'vn_kT_{number_events}_nspec_{n_spec_used}_SENZACUT.png')
filename_v2_pdf = os.path.join(out_dir, f'vn_kT_{number_events}_nspec_{n_spec_used}_SENZACUT.pdf')
plt.savefig(filename_v2, dpi=300)
plt.savefig(filename_v2_pdf)
# plt.show()
#==============================================================================