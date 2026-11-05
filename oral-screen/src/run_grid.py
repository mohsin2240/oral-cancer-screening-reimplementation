"""Run the paper's experiment grid (Tables IV-IX) sequentially, skipping configurations that already finished.

Every run uses data/manifest_train5.csv (training split at the paper's 5:1 imbalance, val/test with every suspicious
image) unless stated otherwise. Each run's output dir is results/<manifest>/<config>/ (see train.default_out).

Usage:
  python src/run_grid.py --groups table4 table5 table6 --seeds 0 1 2 --epochs 50
  python src/run_grid.py --list
Groups: table4 table5 table6 table7 teachers table8 recipe external
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from models import STUDENTS, TEACHERS

MANIFEST = "data/manifest_train5.csv"
EXTERNAL_MANIFEST = "data/manifest_external.csv"

# Per-model memory settings for a 4 GB GPU: (batch, accum) keeps the effective batch at 32.
BATCH = {"efficientnetv2_b0": (16, 2), "mobilenetv3_l": (16, 2), "nasnet_mobile": (16, 2),
         "vit_b16": (16, 2), "swin_b": (16, 2), "effv2_l": (4, 8)}


# The large transformer teachers do not train at our default lr of 1e-3 (test AUROC ~55; the paper gives none) and still collapse at 1e-4, so they get 1e-5.
TEACHER_LR = {"vit_b16": 1e-5, "swin_b": 1e-5}


def teacher_dir(t):
    lr = TEACHER_LR.get(t)
    return Path("results/manifest_train5") / (f"{t}_geometric_wce" + (f"_lr{lr:g}" if lr else ""))


def run(model, **kw):
    b, a = BATCH.get(model, (32, 1))
    args = {"model": model, "batch": b, "accum": a, "manifest": MANIFEST, **kw}
    return args


def experiments():
    g = {}
    # Table IV: augmentation strategies, plain BCE (baseline = none)
    g["table4"] = [run(m, aug=aug) for m in STUDENTS for aug in ["none", "geometric", "zoom", "colour", "sam"]]
    # Table V: image-quality reweighting of the loss
    g["table5"] = [run(m, iqa=q) for m in STUDENTS for q in ["niqe", "clipiqa"]]
    # Table VI: class-imbalance handling
    g["table6"] = [run(m, **kw) for m in STUDENTS for kw in [
        {"sampling": "under"}, {"sampling": "over"}, {"loss": "wce"}, {"loss": "focal"},
        {"mixup_alpha": 0.1}, {"mixup_alpha": 0.2}, {"mixup_alpha": 0.3}]]
    # The final recipe, with checkpoints kept for robustness.py / explain.py
    g["recipe"] = [run("mobilevitv2", aug="geometric", loss="wce", keep_ckpt=True)]
    # Table VII: 5-fold CV of the recipe (train+val pooled; test untouched)
    g["table7"] = [run("mobilevitv2", aug="geometric", loss="wce", fold=f) for f in range(5)]
    # Teachers for Table VIII (one seed each, as the paper trains each teacher once)
    g["teachers"] = [run(t, aug="geometric", loss="wce", keep_ckpt=True, seeds=[0],
                         **({"lr": TEACHER_LR[t]} if t in TEACHER_LR else {})) for t in TEACHERS]
    # Table VIII: distillation into MobileViTv2 at 50% and 100% KD weight
    g["table8"] = [run("mobilevitv2", aug="geometric", loss="wce", teacher=t, kd_beta=b,
                       teacher_ckpt=(teacher_dir(t) / "seed0.pt").as_posix())
                   for t in TEACHERS for b in [0.5, 1.0]]
    # Held-out source: train/val/test on Mendeley, evaluate on the Kaggle source as "external"
    g["external"] = [run("mobilevitv2", manifest=EXTERNAL_MANIFEST),
                     run("mobilevitv2", manifest=EXTERNAL_MANIFEST, aug="geometric", loss="wce")]
    return g


# Rough seconds per epoch on one GPU (measured on an RTX 3050), used only to balance shards.
EPOCH_COST = {"efficientnetv2_b0": 3.2, "mobilenetv3_l": 3.2, "nasnet_mobile": 3.2, "vit_b16": 6.0, "swin_b": 7.0,
              "effv2_l": 36.0}


def cost(item, n_seeds):
    _, out, cli = item
    model = cli[cli.index("--model") + 1]
    c = EPOCH_COST.get(model, 1.8) * (1 if "teachers" == item[0] else n_seeds)
    if "kd-effv2_l" in out.name:
        c += 10.0 * n_seeds  # teacher forward pass at 384 px
    if "mixup" in out.name:
        c *= 1.5
    return c


def shard(todo, k, n, n_seeds):
    """Greedy cost-balanced split. Teachers and the KD runs that need their checkpoints stay together on shard 0,
    in queue order, so a student is never distilled before its teacher exists."""
    loads, assign = [0.0] * n, {}
    for item in todo:
        if item[0] in ("teachers", "table8"):
            assign[item[1]] = 0
            loads[0] += cost(item, n_seeds)
    for item in sorted((t for t in todo if t[1] not in assign), key=lambda t: -cost(t, n_seeds)):
        s = loads.index(min(loads))
        assign[item[1]] = s
        loads[s] += cost(item, n_seeds)
    return [t for t in todo if assign[t[1]] == k]


def run_with_timeout(cli, log, timeout_min):
    """Run one configuration; on timeout kill it together with its DataLoader workers. Returns the exit code or
    'timeout'."""
    posix = os.name == "posix"
    p = subprocess.Popen(cli, stdout=log, stderr=subprocess.STDOUT, start_new_session=posix)
    try:
        return p.wait(timeout=timeout_min * 60 if timeout_min else None)
    except subprocess.TimeoutExpired:
        if posix:
            os.killpg(p.pid, signal.SIGKILL)
        else:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL)
        p.wait()
        return "timeout"


def to_cli(cfg, seeds, epochs, workers):
    cli = [sys.executable, "src/train.py", "--epochs", str(epochs), "--workers", str(workers)]
    cfg = dict(cfg)
    cli += ["--seeds", *map(str, cfg.pop("seeds", seeds))]
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cli.append(flag)
        elif v is not False and v is not None:
            cli += [flag, str(v)]
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["table4", "table5", "table6", "recipe", "table7", "teachers",
                                                    "table8", "external"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="run only this shard (one process per GPU)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--time-budget-h", type=float, default=None,
                    help="stop starting new configurations after this many hours (e.g. Kaggle's 12 h limit)")
    ap.add_argument("--timeout-min", type=float, default=None,
                    help="kill a configuration that runs longer than this (guards against DataLoader deadlocks)")
    args = ap.parse_args()

    from train import default_out

    exps = experiments()
    seen = set()
    queue = []
    for grp in args.groups:
        for cfg in exps[grp]:
            ns = argparse.Namespace(model=cfg["model"], aug=cfg.get("aug", "none"), loss=cfg.get("loss", "bce"),
                                    sampling=cfg.get("sampling", "none"), mixup_alpha=cfg.get("mixup_alpha", 0.0),
                                    iqa=cfg.get("iqa", "none"), teacher=cfg.get("teacher"),
                                    kd_beta=cfg.get("kd_beta", 0.5), fold=cfg.get("fold"), lr=cfg.get("lr", 1e-3),
                                    manifest=Path(cfg["manifest"]))
            out = default_out(ns)
            if out in seen:  # e.g. the BCE/none baseline is shared by Tables IV-VI
                continue
            seen.add(out)
            queue.append((grp, out, to_cli(cfg, args.seeds, args.epochs, args.workers)))

    todo = [q for q in queue if not (q[1] / "summary.json").exists()]
    print(f"{len(queue)} configurations, {len(queue) - len(todo)} already done, {len(todo)} to run")
    if args.num_shards > 1:
        todo = shard(todo, args.shard, args.num_shards, len(args.seeds))
        print(f"shard {args.shard}/{args.num_shards}: {len(todo)} configurations")
    if args.list:
        for grp, out, _ in todo:
            print(f"  [{grp}] {out}")
        return
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i, (grp, out, cli) in enumerate(todo, 1):
        if args.time_budget_h and time.time() - start > args.time_budget_h * 3600:
            print(f"time budget of {args.time_budget_h} h reached; {len(todo) - i + 1} configurations left", flush=True)
            break
        print(f"[{i}/{len(todo)}] [{grp}] {out}", flush=True)
        with open(log_dir / f"{out.parent.name}__{out.name}.log", "w") as f:
            rc = run_with_timeout(cli, f, args.timeout_min)
        print(f"    exit {rc}", flush=True)


if __name__ == "__main__":
    main()
