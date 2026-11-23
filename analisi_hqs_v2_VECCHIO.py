import numpy as np
import glob
import os
import sys
import re
import matplotlib.pyplot as plt

# --- setup cartella dati ---
percorso_file = os.path.abspath(__file__)
data_dir =  os.path.dirname(percorso_file)

# cartella di output per i file del v2 degli heavy quarks
out_dir = os.path.join(data_dir, "v2_hq")
os.makedirs(out_dir, exist_ok=True)

# --- kT binning ---
delta_kT = 0.25
kT_bins = np.arange(0.0, 5.0 + delta_kT, delta_kT)
kT_centers = 0.5 * (kT_bins[:-1] + kT_bins[1:])

# --- reference range for 2PC ---
pref_min, pref_max = 1.0, 6.0   # GeV

# --------- utilities ----------
def extract_number(f):
    m = re.search(r"(\d+)", f)
    return int(m.group(1)) if m else -1

def calculate_v2_2pc_event(kx, ky, w, kT_bins, n=2, pref_min=1.0, pref_max=6.0):
    """
    Compute v_n{2PC} for a single event, excluding self-correlations.
    Inputs:
      kx, ky : arrays of momenta
      w      : weights (here = 1 per particle)
      kT_bins: bin edges in kT
      n      : harmonic order (default 2)
    Returns:
      vn_2pc (per-bin), VnDelta_refref
    """
    kT = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)

    nbins = len(kT_bins) - 1
    Qn_bins = np.zeros(nbins, dtype=complex)
    N_bins  = np.zeros(nbins)

    # build Qn for each kT bin
    for i, (k_lo, k_hi) in enumerate(zip(kT_bins[:-1], kT_bins[1:])):
        mask = (kT >= k_lo) & (kT < k_hi)
        if np.any(mask):
            w_sel = w[mask]
            phi_sel = phi[mask]
            Qn_bins[i] = np.sum(w_sel * np.exp(1j * n * phi_sel))
            N_bins[i] = np.sum(w_sel)

    VnDelta = np.full(nbins, np.nan)

    for i in range(nbins):
        if N_bins[i] > 0:
           
            mask_ref = (kT >= pref_min) & (kT <= pref_max) 

            if not np.any(mask_ref):
                continue  # no reference particles left after excluding bin i

            w_ref = w[mask_ref]
            phi_ref = phi[mask_ref]
            Qn_ref = np.sum(w_ref * np.exp(1j * n * phi_ref))
            N_ref  = np.sum(w_ref)

            if N_ref > 0:

                k_bin_center = 0.5 * (kT_bins[i] + kT_bins[i+1])

                if pref_min <= k_bin_center <= pref_max:
                # bin is inside the reference window → exclude self-correlations
                    VnDelta[i] = (np.real(Qn_bins[i] * np.conj(Qn_ref)) - N_bins[i]) / (N_bins[i] * (N_ref - 1))

                else:
            # bin is outside reference window → normal calculation
                    VnDelta[i] = np.real(Qn_bins[i] * np.conj(Qn_ref)) / (N_bins[i] * N_ref)


                

    # --- global ref-ref (full pref range, no exclusion) ---
    mask_ref_global = (kT >= pref_min) & (kT <= pref_max)
    if not np.any(mask_ref_global):
        return np.full(nbins, np.nan), np.nan

    w_ref_g = w[mask_ref_global]
    phi_ref_g = phi[mask_ref_global]
    Qn_ref_g = np.sum(w_ref_g * np.exp(1j * n * phi_ref_g))
    N_ref_g  = np.sum(w_ref_g)
    VnDelta_refref = (np.real(Qn_ref_g * np.conj(Qn_ref_g))-N_ref_g) / (N_ref_g * (N_ref_g-1))

    vn_2pc = VnDelta / np.sqrt(VnDelta_refref) if VnDelta_refref > 0 else np.full(nbins, np.nan)

    print(f"Event / Evento: VnDelta_refref = {VnDelta_refref:.6f}, N_ref_g = {N_ref_g}")
    

    return vn_2pc, VnDelta_refref

# --------- main ----------

files = sorted(glob.glob(os.path.join(data_dir, "HQinitializations/XandP_tfin_*.csv")), key=extract_number)
if not files:
    print("No files found matching XandP_tfin_*.csv")
    sys.exit()

vn_all = []
for i, filename in enumerate(files):
    data = np.genfromtxt(filename, delimiter=',')
    kx = data[:,2]
    ky = data[:,3]

    w  = np.ones_like(kx)   # one weight per particle (just counting)

    vn_evt, V2_refref = calculate_v2_2pc_event(
        kx, ky, w, kT_bins, n=2, pref_min=pref_min, pref_max=pref_max
    )
    if not np.all(np.isnan(vn_evt)):
        vn_all.append(vn_evt)

    print(f"Processed {i}/{len(files)}: {filename}")



vn_all = np.array(vn_all)
if vn_all.size == 0:
    print("No valid events for 2PC in the chosen reference range!")
    sys.exit()



vn_mean = np.nanmean(vn_all, axis=0)
vn_err  = np.nanstd(vn_all, axis=0, ddof=1) / np.sqrt(len(vn_all))
print(f"Number of valid events per bin / Numero eventi validi per bin: {np.sum(~np.isnan(vn_all), axis=0)}")

# save results
out = np.column_stack((kT_centers, vn_mean, vn_err))
outname = os.path.join(out_dir, f"v2_2PC_no_auto_results_{len(files)}events.txt")
header = "kT[GeV], v2{2PC}, stat.err"
np.savetxt(outname, out, delimiter=" ", header=header, fmt="%.6f")
print(f"v2 2PC results saved to {outname}")

# --------- plotting v2 vs kT ---------
file2 = os.path.join(out_dir, f"v2_2PC_no_auto_results_{len(files)}events.txt")
data2 = np.genfromtxt(file2)
x = data2[:,0]
v2_avg = data2[:,1]
v2_std = data2[:,2]

plt.errorbar(kT_centers, vn_mean, yerr=vn_err, fmt='o', capsize=3, label=r'$v_2\{2PC\}$')
plt.plot(x, v2_avg, label="v₂ EP", color="blue")
plt.fill_between(x, v2_avg - v2_std, v2_avg + v2_std, color="blue", alpha=0.3, label="v₂ ± σ EP")
plt.xlabel(r"$k_T$ [GeV]")
plt.ylabel(r"$v_2$")
plt.title(f"v2 comparison from {len(files)} events")
plt.legend()
plt.grid(True)
#plt.show()
plotname = os.path.join(out_dir, f"v2_2PC_results_{len(files)}events.png")
plt.savefig(plotname, dpi=300)
plotname2 = os.path.join(out_dir, f"v2_2PC_results_{len(files)}events.pdf")
plt.savefig(plotname2)
#plt.close()

print(f"Plot saved to {plotname}")
