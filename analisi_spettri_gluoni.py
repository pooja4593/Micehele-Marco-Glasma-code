# ───────────────────────────────────────────────────────────────
# 1) Parametri e binning ricavati dai dati
# ───────────────────────────────────────────────────────────────
import numpy as np, pandas as pd, glob, os
import matplotlib
matplotlib.use('Agg')  # usa backend non interattivo

import matplotlib.pyplot as plt

import glasma_config as cfg
import sys

Lx, Nx = cfg.Lx, cfg.Nx
a_fm       = Lx / Nx                  # [fm]
fm_to_GeV  = 5.067730937              # 1 fm = 5.06773 GeV⁻¹
a          = a_fm * fm_to_GeV         # lattice spacing [GeV⁻¹]

delta_kT   = 0.25                     # larghezza del bin (GeV, lattice)
files      = sorted(glob.glob('spectrum_event/spectrum_event[0-9]*.csv'))

n_spec_used = cfg.i_cho_spe_anal


# ––– guardo un file per stimare il massimo kT di lattice davvero presente
df_tmp = pd.read_csv(
    files[0],
    names=['kx','ky','dNd2k_FULL', 'dNd2k_noODD', 'dNd2k_EQUIPA'],
    sep=',',            # oppure sep=r'\s*,\s*' se hai spazi
    skipinitialspace=True # ignora lo spazio dopo la virgola
    ).astype(float)


kT_max_data = np.sqrt(df_tmp.kx.values**2 + df_tmp.ky.values**2).max()


# arrotondo al passo di bin (o impongo un valore a mano se preferisci)
kT_max = np.ceil(kT_max_data / delta_kT) * delta_kT



IR_lim = 0.25 # IR cutof per calcolo spettro
kT_bins    = np.arange(IR_lim, kT_max + delta_kT, delta_kT)
kT_centres = 0.5 * (kT_bins[1:] + kT_bins[:-1])
# ───────────────────────────────────────────────────────────────



# ───────────────────────────────────────────────────────────────
# 2) Funzioni ausiliarie
# ───────────────────────────────────────────────────────────────
def spectrum_single_event(df, kT_bins, n_spec):
    """
    Restituisce array con dN/(2π kT dkT) in ciascun bin radiale.
    """
    kx_t = df.kx.values          # già impulsi di lattice
    ky_t = df.ky.values
    kT   = np.sqrt(kx_t**2 + ky_t**2)
    
    if n_spec==1:
        dNd2k = df.dNd2k_FULL.values

    elif n_spec==2:
        dNd2k = df.dNd2k_noODD.values

    elif n_spec==3:
        dNd2k = df.dNd2k_EQUIPA.values

    spec = np.full(len(kT_centres), np.nan)
    for i, (k_lo, k_hi) in enumerate(zip(kT_bins[:-1], kT_bins[1:])):
        mask = (kT >= k_lo) & (kT < k_hi)
        if mask.any():
            spec[i] = dNd2k[mask].mean()
    return spec

# ───────────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────────
# 3) Main: calcola spettro per tutti gli eventi e la media
# ───────────────────────────────────────────────────────────────
spectra_evt = []
for f in files:
    df = pd.read_csv(
        f,
        names=['kx','ky','dNd2k_FULL', 'dNd2k_noODD', 'dNd2k_EQUIPA'],
        sep=',',              # separatore “,”
        skipinitialspace=True # ignora lo spazio dopo la virgola
        ).astype(float)

    print(f)
    spectra_evt.append(spectrum_single_event(df, kT_bins, n_spec_used))
 
 
spectra_evt = np.array(spectra_evt)        # shape = (N_evt, N_kT)

# media sugli eventi, errore statistico punto-per-punto
spec_mean = np.nanmean(spectra_evt, axis=0)
spec_err  = np.nanstd (spectra_evt, axis=0, ddof=0) / \
            np.sqrt(np.count_nonzero(~np.isnan(spectra_evt), axis=0))

# salva risultati
out = pd.DataFrame({'kT_latt[GeV]':kT_centres,
                    'dN_over_2pi_kT_dkT':spec_mean,
                    'stat_err':spec_err})
out.to_csv('spectrum_radial_average.csv',
           index=False,
           header=False)      # ← niente riga di header


print("Average radial spectrum saved to / Spettro radiale medio salvato in  spectrum_radial_average.csv")
print(out.head())
# ───────────────────────────────────────────────────────────────




# ───────────────────────────────────────────────────────────────
# 5) Plot log-log + salvataggio
# ───────────────────────────────────────────────────────────────

plt.figure()

# assicuriamoci di usare i soli bin con dati validi
mask = ~np.isnan(out['dN_over_2pi_kT_dkT'])
kT_plot   = out['kT_latt[GeV]'][mask]
spec_plot = out['dN_over_2pi_kT_dkT'][mask]
err_plot  = out['stat_err'][mask]

# errorbar con assi logaritmici
# plt.errorbar(kT_plot, spec_plot * kT_plot**4, xerr=delta_kT/2, yerr=err_plot,
#              fmt='o', markersize=1, capsize=3, elinewidth=1)
plt.errorbar(kT_plot, spec_plot , xerr=delta_kT/2,  
             fmt='o', markersize=1, capsize=3, elinewidth=1,
             label = r'$dN/2\pi k_T dk_T$')

norm2 = 1e-2
norm3 = 3e-2
norm4 = 4e-1

f2 = norm2 / kT_plot**2
f3 = norm3 / kT_plot**3
f4 = norm4 / kT_plot**4

lw = 0.5

 





plt.xscale('linear')
plt.yscale('log')
# plt.xlim(6e-1,2e1)
# plt.ylim(1e-5,2e3)
plt.xlabel(r'$k_T^{\mathrm{latt}}\;[\mathrm{GeV}]$')
plt.ylabel(r'$dN / \left(2\pi\,k_T \, dk_T\right)$')
plt.title(r'128x128, pA, $N_A=4$, $\tau=0.35$ fm')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend()

plt.tight_layout()

# salva entrambe le versioni
plt.savefig(f'spectrum_radial_average_{n_spec_used}.png', dpi=300)
plt.savefig(f'spectrum_radial_average_{n_spec_used}.pdf')

# plt.show()

