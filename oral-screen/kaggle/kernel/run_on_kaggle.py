"""Kaggle notebook entry point: runs (part of) the experiment grid on GPU T4 x2.

Expects the private dataset "<user>/oral-screen-bundle" (built by kaggle/make_bundle.py) attached. Runs two grid
shards in parallel, one per T4, then the noise stress test, explainability and the tables, and finally packs everything
into /kaggle/working/results.zip.

Safeguards learned from the first session:
- A Kaggle version that hits the 12 h limit is discarded together with its outputs, so new configurations stop
  starting after TIME_BUDGET_H and any single configuration is killed after TIMEOUT_MIN.
- Two DataLoaders with worker processes (Balanced Mix-up) deadlocked on Kaggle's 4 vCPUs, so data is loaded in the
  main process (WORKERS = 0); the corpus is small and cached in memory.
- /kaggle/working keeps only ~500 files, so results are zipped into a single archive.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

GROUPS = "table4 table5"  # session A: "recipe teachers table8 table7 external table6"
SEEDS = "0 1 2 3 4"
EPOCHS = "50"
WORKERS = "0"
TIME_BUDGET_H = 9.5
TIMEOUT_MIN = 90

# Locate the attached dataset (the mount path differs between Kaggle versions).
bundle = next(os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/src/run_grid.py", recursive=True))
work = "/kaggle/oral-screen"  # outside /kaggle/working so intermediate files don't count against the output limit
shutil.copytree(bundle, work, dirs_exist_ok=True)
os.chdir(work)
print("bundle:", bundle, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam", "tabulate"], check=False)
subprocess.run(["nvidia-smi"], check=False)

procs = []
for gpu in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [sys.executable, "src/run_grid.py", "--groups", *GROUPS.split(), "--seeds", *SEEDS.split(),
           "--epochs", EPOCHS, "--workers", WORKERS, "--num-shards", "2", "--shard", str(gpu),
           "--time-budget-h", str(TIME_BUDGET_H), "--timeout-min", str(TIMEOUT_MIN)]
    log = open(f"grid_shard{gpu}.log", "w")
    procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    time.sleep(30)  # stagger the first model downloads

while any(p.poll() is None for p, _ in procs):
    time.sleep(300)
    for gpu in (0, 1):
        with open(f"grid_shard{gpu}.log") as f:
            lines = [l.strip() for l in f if l.startswith(("[", "    exit", "time budget"))]
        print(f"[{time.strftime('%H:%M')}] shard {gpu}: {' | '.join(lines[-2:]) if lines else 'starting'}", flush=True)
for p, log in procs:
    log.close()
    print("shard exit", p.returncode, flush=True)

recipe = "results/manifest_train5/mobilevitv2_geometric_wce"
if os.path.exists(f"{recipe}/seed0.pt") and not os.path.exists(f"{recipe}/robustness.csv"):
    for script in ("robustness.py", "explain.py"):
        subprocess.run([sys.executable, f"src/{script}", "--run-dir", recipe, "--manifest", "data/manifest_train5.csv"])
subprocess.run([sys.executable, "src/report.py"])

for gpu in (0, 1):
    shutil.copy(f"grid_shard{gpu}.log", "results/logs/")
shutil.make_archive("/kaggle/working/results", "zip", work, "results")
print("wrote /kaggle/working/results.zip", os.path.getsize("/kaggle/working/results.zip") // 2**20, "MB", flush=True)
