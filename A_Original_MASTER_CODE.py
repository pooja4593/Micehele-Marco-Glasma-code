import os



# questo svuota il file nohup.out in caso ci sia nella cartella
try:
    open("nohup.out", "w").close()
except Exception:
    pass




import sys, shutil

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True





# 🔥 Elimina tutte le cartelle __pycache__ ricorsivamente
for root, dirs, files in os.walk(".", topdown=False):
    for d in dirs:
        if d == "__pycache__":
            pycache_path = os.path.join(root, d)
            print(f"🧹 Rimuovo cartella cache: {pycache_path}")
            shutil.rmtree(pycache_path, ignore_errors=True)
            
          

import subprocess, glob, time

from joblib import Parallel, delayed
import multiprocessing
from glasma_config import NRUNTOT, i_cho_spe_anal, HQUARKS, email_send
 

# ========== CRONOMETRO INIZIA ==========
start_time = time.time()
print("\n⏳ Avvio esecuzione completa...\n" + "="*60)

# ========== PULIZIA ==========
print("\n🧹 Pulizia pre-run...\n" + "="*60)
for f in glob.glob("theta_event*.txt"): os.remove(f)

for f in glob.glob("log*.txt"):
    print(f"  - Rimuovo file: {f}")
    os.remove(f)
    
for ext in ["*.log", "*.pdf", "*.png", "*.csv", "*.dat"]:
    for f in glob.glob(ext):
        print(f"  - Rimuovo file: {f}")
        os.remove(f)

# file .txt prodotti dagli script di analisi v2 degli heavy quarks
# (analisi_hqs_v2_VECCHIO.py e analisi_hqs_v2_oct26_def.py).
# Ora questi script scrivono in v2_hq/: il blocco serve solo a eliminare
# eventuali file rimasti nella cartella principale da run precedenti.
for pattern in ["v2_2PC_no_auto_results_*events.txt",   # VECCHIO
                "VnDelta_refref_media_*.txt",           # oct26_def
                "VnDelta_refref_per_evento_*.txt",      # oct26_def
                "VnDelta_vs_kT_*.txt",                  # oct26_def
                "v2_rapportoMedie_*.txt"]:              # oct26_def
    for f in glob.glob(pattern):
        print(f"  - Rimuovo file: {f}")
        os.remove(f)

if os.path.exists("evolutionaverage"):
    print("  - Rimuovo cartella: evolutionaverage")
    shutil.rmtree("evolutionaverage")


for folder in glob.glob("evolution_run*"):
    if os.path.isdir(folder):
        print(f"  - Rimuovo cartella: {folder}")
        shutil.rmtree(folder)

if os.path.exists("spectrum_event"):
    print("  - Rimuovo cartella: spectrum_event")
    shutil.rmtree("spectrum_event")

# cartelle di output del vn dei gluoni e del v2 degli heavy quarks
for folder in ["vn_gluoni", "v2_hq"]:
    if os.path.exists(folder):
        print(f"  - Rimuovo cartella: {folder}")
        shutil.rmtree(folder)
        
log_run_dir = "log_run_events"
if os.path.exists(log_run_dir):
    print(f"  - Rimuovo cartella: {log_run_dir}")
    shutil.rmtree(log_run_dir)


        

hq_init_dir = "HQinitializations"
if os.path.exists(hq_init_dir):
    for f in os.listdir(hq_init_dir):
        file_path = os.path.join(hq_init_dir, f)
        if os.path.isfile(file_path):  # assicura che sia un file, non una subdirectory
            print(f"  - Rimuovo file: {f}")
            os.remove(file_path)


anal_init_dir = "Analysis"
if os.path.exists(anal_init_dir):
    for f in os.listdir(anal_init_dir):
        file_path = os.path.join(anal_init_dir, f)
        if os.path.isfile(file_path):  # assicura che sia un file, non una subdirectory
            print(f"  - Rimuovo file: {f}")
            os.remove(file_path)

        
print("✅ Pulizia completata\n" + "="*60)

# ========== PARALLELISMO INCORPORATO ==========
# def run_glasma(nrun):
#     subprocess.run([sys.executable, "glasma_main_whqs_16luglio2025.py",
#                     str(nrun), str(NRUNTOT)])
    
    
def run_glasma_with_log(nrun):

    # Crea la cartella se non esiste
    log_dir = "log_run_events"
    os.makedirs(log_dir, exist_ok=True)

    # File di log per il singolo run
    log_filename = os.path.join(log_dir, f"run_{nrun}.log")
    print(f"▶️ [RUN {nrun}] Avvio... → log: {log_filename}")



    with open(log_filename, "w") as logfile:
        process = subprocess.Popen(
            [sys.executable, "MA_glasma_evolution_06nov26_def.py", str(nrun), str(NRUNTOT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(f"[{nrun:04d}] {line}")
            logfile.write(line)
        process.wait()
  

def run_parallel_glasma():
    total_cores = multiprocessing.cpu_count()
    NCORES = max(1, total_cores - 2)
    NCORES = min(NCORES, NRUNTOT)
    print(f"[INFO] Core logici totali: {total_cores}")
    print(f"[INFO] Numero di core usati per il parallelo: {NCORES}")

    Parallel(n_jobs=NCORES)(
        delayed(run_glasma_with_log)(nrun) for nrun in range(NRUNTOT)
    )
    print("\n✅ Completato: glasma_main_parallel\n" + "="*60)


# === Crea cartella per le inizializzazioni di U_totx U_toty ed E_eta ===
os.makedirs(f"inizializzazioni_campi_gluoni", exist_ok=True)




run_parallel_glasma()

# ========== FASI DI ANALISI ==========
def run_and_log(script_name, log_name):
    print(f"\n▶️ Lancio {script_name}...\n{'='*60}")
    with open(log_name, "w") as logfile:
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            logfile.write(line)
        process.wait()

run_and_log("analisi_obs_glasma.py", "analisi.log")

# 3. Contourplot Tp1 e Tp2
# run_and_log("contplotTp.py", "contplotTp.log")



hquarks_trigger = HQUARKS
if hquarks_trigger.strip().lower() == "yes":
    run_and_log("analisi_hqs.py", "analisiHQs.log")
    run_and_log("analisi_hqs_finale.py", "analisiHQs.log")
    run_and_log("analisi_hqs_v2_VECCHIO.py", "analisiHQsV2.log")
    #run_and_log("analisi_vn_hqs_epprima.py", "analisiHQsV2EPprima.log")
    #run_and_log("analisi_vn_hqs_epdopo.py", "analisiHQsV2EPdopo.log")

run_and_log("analisi_spettri_gluoni.py", "analisi_spettri_gluoni.log")
run_and_log("analisi_vnep_gluoni.py", "analisi_vnep_gluoni.log")
 


 

     

# ========== TEMPO TOTALE ==========
elapsed = time.time() - start_time
print(f"\n🏁 Fine. Tutto completato in {int(elapsed // 60)} minuti e {int(elapsed % 60)} secondi.")

#------------------------------------------------------------------------------
# MANDA EMAIL
# Optional: if email_sender = yes in glasma_config.py, then the orchestrator
# sends an email at the end of the run
#------------------------------------------------------------------------------





# if email_send.strip().lower() == "yes":
#     from pathlib import Path
#     env_path = Path(".env")
    
    
#     if not env_path.is_file():
#         print("⚠️ ATTENZIONE: Il file .env non è presente nella cartella corrente.")
#         print("   L'invio della mail è stato disabilitato per evitare errori.")
#         print("   Per abilitare l'invio, crea un file .env con le variabili SMTP_USER e SMTP_PASS.")
#         # Esci dal programma o continua senza invio email, a seconda di cosa preferisci:
#         # Per uscire con errore:
#         import sys
#         sys.exit(1)

#     npernome = NRUNTOT
#     spePERnome = i_cho_spe_anal
#     hquarks_trigger = HQUARKS
    
#     import smtplib
#     from email.mime.multipart   import MIMEMultipart
#     from email.mime.text        import MIMEText
#     from email.mime.application import MIMEApplication
#     from dotenv import load_dotenv
#     from pathlib import Path
#     import platform
    
#     hostname = platform.node()
    
#     if "grnode" in hostname:
#         origin_info = "Email inviata dal CLUSTER (grnode01)"
#     else:
#         origin_info = "Email inviata da PC"
    
#     load_dotenv()
    
    
    
#     SMTP_SERVER = "smtp.libero.it"
#     SMTP_PORT   = 465                       # SSL; usa 587 + STARTTLS se 465 desse problemi
#     SMTP_USER   = os.environ["SMTP_USER"]   # glasma.sender@libero.it
#     SMTP_PASS   = os.environ["SMTP_PASS"]    
    
#     def manda_email_avviso(titolo, corpo, destinatario, allegati=None):
#         """
#         Spedisce un'e-mail (anche con allegati) tramite l’account Libero
#         configurato nel file .env
#         """
#         # 1) busta
#         msg            = MIMEMultipart()
#         msg["Subject"] = titolo
#         msg["From"]    = SMTP_USER
#         msg["To"]      = destinatario
#         msg.attach(MIMEText(corpo))
    
#         # 2) allegati (PNG, PDF, txt, …)
#         if allegati:
#             for path in allegati:
#                 with open(path, "rb") as f:
#                     part = MIMEApplication(f.read(),
#                                            Name=os.path.basename(path))
#                 part["Content-Disposition"] = (
#                     f'attachment; filename="{os.path.basename(path)}"'
#                 )
#                 msg.attach(part)
    
#         # 3) invio via SMTP SSL
#         with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
#             server.login(SMTP_USER, SMTP_PASS)
#             server.send_message(msg)
    
#         print("📧  Email inviata con successo!")
    
#     # destinatario = "marco.ruggieri.ufs@gmail.com, marco.ruggieri@dfa.unict.it"
    
#     destinatario = "marco.ruggieri@dfa.unict.it, gabriele.parisi@dfa.unict.it, fabrizio.murgana@dfa.unict.it"
    
#     elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
#     msg_time = (
#         f"🏁 Fine. Tutto completato in "
#         f"{elapsed_min} minuti e {elapsed_sec} secondi."
#     )
    
#     from glasma_config import tf, Npart_p, Npart_A, i_homogen_p, i_homogen_A
    
    
#     OGGETTO=f"⚛️ Collisione {Npart_p} -{ Npart_A} a tau = {tf} fm – {NRUNTOT} eventi"
    
#     corpo_email = (
#         "Ciao Marco,\n\n"
#         "Ciao Gabri,\n\n"
#         "Ciao Murgazzo Gianfranco,\n\n"
#         "ATTENZIONE: risultati SENZA campo magnetico,\n\n"
#         f"{msg_time}\n\n"
#         f"Parametri run:\n"
#         f"  Numero di eventi = {NRUNTOT}\n"
#         f"  tempo finale = {tf} fm\n"
#         f"  Npart_1 = {Npart_p}, i_homogen_1 = {i_homogen_p}\n"
#         f"  Npart_2 = {Npart_A}, i_homogen_2 = {i_homogen_A}\n"
#         f"  {origin_info}\n\n\n"
#         "\n Il lavoro duro paga sempre🪄"
#     )
    
#     png_name = f"vn_kT_{npernome}_nspec_{spePERnome}.png"
#     png_average = [
#         "evolutionaverage/ave_glasma_press.png",
#         f"spectrum_radial_average_{spePERnome}.png",
#         "HQinitializations/HQs_spectrum_radial.png",
#         "HQinitializations/HQs_spectrum_radial_final.png",
#         f"v2_2PC_results_{NRUNTOT}events.png"
#     ]
#     allegati = [png_name] + png_average 
#     allegati = [str(p) for p in map(Path, allegati) if p.is_file()]
    
#     manda_email_avviso(
#         titolo=OGGETTO,
#         corpo=corpo_email,
#         destinatario=destinatario,
#         allegati=allegati
#     )
    
# else:
#     print("⚠️ Invio email disabilitato da configurazione.")
    
