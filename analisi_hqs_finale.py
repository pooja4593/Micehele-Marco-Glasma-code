# ───────────────────────────────────────────────────────────────
# 1) Import ed estrazione parametri
# ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # usa backend non interattivo

import matplotlib.pyplot as plt

import glob, os
import glasma_config as cfg
from scipy.integrate import quad

# Parametri globali da cfg
NRUN = cfg.NRUNTOT
NHQ = cfg.NHQ
hq_flavor = cfg.hq_flavor

delta_pT = 0.25  # larghezza dei bin in GeV

# Limiti IR/UV da settare (eventualmente potremmo ricavarli dai dati)
pT_max = 10.0
pT_min = 0.25

pT_bins    = np.arange(pT_min, pT_max + delta_pT, delta_pT)
pT_centres = 0.5 * (pT_bins[1:] + pT_bins[:-1])

# ───────────────────────────────────────────────────────────────
# 2) Funzione spettro per singolo evento
# ───────────────────────────────────────────────────────────────
def spectrum_HQ_single_event(px, py, pT_bins):
    pT = np.sqrt(px**2 + py**2)
    dNdpT = np.zeros(len(pT_centres))

    for i, (p_lo, p_hi) in enumerate(zip(pT_bins[:-1], pT_bins[1:])):
        mask = (pT >= p_lo) & (pT < p_hi)
        dNdpT[i] = mask.sum()

    # normalizza a dN / (2π pT dpT)
    # dNdpT /= (2 * np.pi * pT_centres * delta_pT)
    dNdpT /= (delta_pT)
    return dNdpT

# ───────────────────────────────────────────────────────────────
# 3) Loop su tutti gli eventi
# ───────────────────────────────────────────────────────────────
spectra_evt = []

for nrun in range(NRUN):
    fileHQ = f"HQinitializations/XandP_tfin_{nrun}.csv"
    if not os.path.exists(fileHQ):
        print(f"WARNING: file {fileHQ} not found, skipping / ATTENZIONE: file {fileHQ} non trovato, salto...")
        continue

    data = np.loadtxt(fileHQ, delimiter=",")  # colonne: x, y, px [GeV], py [GeV]
    px = data[:, 2]
    py = data[:, 3]

    dNdpT_evt = spectrum_HQ_single_event(px, py, pT_bins)
    spectra_evt.append(dNdpT_evt)

spectra_evt = np.array(spectra_evt)

# ───────────────────────────────────────────────────────────────
# 4) Calcolo media ed errore statistico
# ───────────────────────────────────────────────────────────────
spec_mean = np.nanmean(spectra_evt, axis=0)
spec_err  = np.nanstd (spectra_evt, axis=0, ddof=0) / \
            np.sqrt(np.count_nonzero(~np.isnan(spectra_evt), axis=0))

# Salva su CSV
out = pd.DataFrame({'pT[GeV]': pT_centres,
                    'dN/dpT': spec_mean,
                    'stat_err': spec_err})
out.to_csv("HQs_spectrum_radial_final.csv", index=False, header=False)

print("Average HQ spectrum saved to / Spettro HQ medio salvato in  HQs_spectrum_radial_final.csv")
print(out.head())


NTOT_HQs = np.sum(spec_mean) * delta_pT
# print(f"N HQs da spettro integrato = {NTOT_HQs}")
# print(f"N HQs da input = {NHQ}")
# print(f"Errore relativo: {100 * np.abs(NHQ - NTOT_HQs) / NHQ}%")





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
     raise ValueError("Invalid flavor. Use 'charm' or 'beauty' / Flavor non valido. Usa 'charm' o 'beauty'.")

 # Funzione di densità non normalizzata
def fonll_pt_dist(pt):
     return pt * x0 / (1 + x3 * pt**x1)**x2

# Integrazione numerica per ottenere la normalizzazione
pT_min = 0.01
pT_max = 10.0
Norm, err = quad(fonll_pt_dist, pT_min, pT_max)
print(f"FONLL integral up to / Integrale FONLL fino a {pT_max} = {Norm:.4f}")

pT_min_test = 0.01
pT_max_test = 5.0
NormTEST, errTEST = quad(fonll_pt_dist, pT_min_test, pT_max_test)
print(f"FONLL integral up to / Integrale FONLL fino a {pT_max_test} = {NormTEST:.4f}")

# Funzione normalizzata a NHQ
def fonll_pt_dist_NORM(pt):
    return NHQ * fonll_pt_dist(pt) / Norm

# ───────────────────────────────────────────────────────────────
# 5) Plot spettro log-log
# ───────────────────────────────────────────────────────────────
# plt.figure()

# mask = ~np.isnan(spec_mean)
# plt.errorbar(pT_centres[mask], spec_mean[mask],
#              xerr=delta_pT/2, yerr=spec_err[mask],
#              fmt='o', markersize=1.5, capsize=3, elinewidth=1,
#              label=r'$dN/dp_T$')

# plt.xscale('linear')
# plt.yscale('linear')

# plt.xlabel(r'$p_T$ [GeV]')
# plt.ylabel(r'$dN / dp_T$')
# plt.title(f'Spettro iniziale {cfg.hq_flavor}, HQs: {NHQ}')
# plt.grid(True, which='both', ls='--', alpha=0.4)
# plt.legend()
# plt.tight_layout()

# plt.savefig("HQinitializations/HQs_spectrum_radial_EB.png", dpi=300)
# plt.savefig("HQinitializations/HQs_spectrum_radial_EB.pdf")
# plt.show()

# === Colore principale ===
col = 'darkblue'   

plt.figure(figsize=(6.5, 5))

# === Banda di errore ±1σ ===
plt.fill_between(pT_centres,
                 (spec_mean - spec_err) ,
                 (spec_mean + spec_err) ,
                 color=col,
                 alpha=0.3,
                 label="Errore statistico")


pt_vals = np.linspace(pT_min, pT_max, 300)
dist_vals = fonll_pt_dist_NORM(pt_vals)

plt.plot(pt_vals, dist_vals , label="FONLL normalizzata", color='orange')

plt.plot(pT_centres, spec_mean , color=col, label=r"$dN/dp_T$", lw=2)



plt.xlabel(r"$p_T$ [GeV]", fontsize=13)
plt.ylabel(r"$dN/dp_T$", fontsize=13)
plt.title(f'Spettro finale {cfg.hq_flavor}, HQs: {NHQ}')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig("HQinitializations/HQs_spectrum_radial_final.png", dpi=300)
plt.savefig("HQinitializations/HQs_spectrum_radial_final.pdf")

# plt.show()
 