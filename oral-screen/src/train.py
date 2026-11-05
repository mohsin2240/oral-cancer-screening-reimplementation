"""Train / validate / test one configuration over several seeds (Sections II-C and III).

Per seed: train for --epochs, keep the checkpoint with the best validation sensitivity x specificity, evaluate it on the
test split (and the external split if the manifest has one). Across seeds: mean +- std, and the paper's "best pair" =
(sens, spec) of the seed with the highest test geometric mean.

Ablation switches (one at a time, as in the paper's tables):
  --aug {none,geometric,zoom,colour,sam}       Table IV
  --iqa {none,niqe,clipiqa}                    Table V   (needs data/iqa_scores.csv from iqa_scores.py)
  --sampling {none,under,over}, --loss {bce,wce,focal}, --mixup-alpha A     Table VI
  --fold k --n-folds 5                         Table VII (train+val pooled and re-split; test untouched)
  --teacher KEY --teacher-ckpt PATH --kd-beta B                             Table VIII

Example:
  python src/train.py --manifest data/manifest_train5.csv --model mobilevitv2 --aug geometric --loss wce
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import OralDataset, build_transform, load_manifest
from iqa_scores import iqa_weights
from losses import build_loss, kd_loss
from models import build_model, count_params, resize_to
from samplers import balanced_mixup, class_balanced_sampler, resample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y, prob, thr=0.5, prevalence=None):
    """prevalence: if given, also report NPV/PPV re-weighted to that suspicious-class prevalence (Bayes), since our
    val/test class ratio differs from the paper's screening population."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    m = {
        "sens": sens,
        "spec": spec,
        "gm": float(np.sqrt(sens * spec)),
        "prod": sens * spec,
        "npv": tn / max(tn + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "acc": (tp + tn) / max(len(y), 1),
        "auroc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if prevalence:
        p = prevalence
        m["npv_adj"] = spec * (1 - p) / max(spec * (1 - p) + (1 - sens) * p, 1e-12)
        m["ppv_adj"] = sens * p / max(sens * p + (1 - spec) * (1 - p), 1e-12)
    return m


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, ys = [], []
    for x, y, _ in loader:
        with torch.autocast(device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def make_loader(ds, batch, workers, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle and sampler is None, sampler=sampler, num_workers=workers,
                      pin_memory=True, drop_last=drop_last, persistent_workers=workers > 0)


def get_splits(args):
    df = load_manifest(args.manifest)
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test", "external"]}
    if args.fold is not None:
        pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
        skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=42)
        tr_idx, va_idx = list(skf.split(pool, pool["label"]))[args.fold]
        splits["train"], splits["val"] = pool.iloc[tr_idx].reset_index(drop=True), pool.iloc[va_idx].reset_index(drop=True)
    if args.max_train:
        splits["train"] = splits["train"].sample(min(args.max_train, len(splits["train"])), random_state=0)
    return {k: v for k, v in splits.items() if len(v)}


def load_teacher(args, device):
    teacher, tspec = build_model(args.teacher, pretrained=False)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu"))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, tspec.input_size


def run_seed(args, seed, splits, device, out_dir: Path, teacher=None):
    set_seed(seed)
    model, spec = build_model(args.model, pretrained=not args.scratch)
    model.to(device)
    size = args.input_size or spec.input_size
    train_tf = build_transform(size, args.aug, train=True)
    eval_tf = build_transform(size)

    tr = resample(splits["train"], args.sampling, seed)
    n_pos = int(tr["label"].sum())
    n_neg = len(tr) - n_pos
    criterion = build_loss(args.loss, n_neg, n_pos)
    weights = iqa_weights(pd.read_csv(args.iqa_scores), tr["id"], args.iqa) if args.iqa != "none" else None
    mask_dir = args.data_root / "sam_masks" if args.aug == "sam" else None
    train_ds = OralDataset(tr, args.data_root, train_tf, weights, mask_dir, args.sam_prob)
    loaders = {"train": make_loader(train_ds, args.batch, args.workers, shuffle=True, drop_last=len(tr) > args.batch)}
    if args.mixup_alpha:
        sampler = class_balanced_sampler(tr["label"], len(tr), seed)
        loaders["balanced"] = make_loader(train_ds, args.batch, args.workers, sampler=sampler, drop_last=len(tr) > args.batch)
    eval_names = [k for k in splits if k != "train"]
    for k in eval_names:
        loaders[k] = make_loader(OralDataset(splits[k], args.data_root, eval_tf), args.batch * 2, args.workers)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5, min_lr=1e-7)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    ckpt = out_dir / f"seed{seed}.pt"
    best_prod, best_loss, best_epoch, log = -1.0, float("inf"), -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        batches = zip(loaders["train"], loaders["balanced"]) if args.mixup_alpha else ((b, None) for b in loaders["train"])
        n_steps = len(loaders["train"])
        for step, (batch, bal) in enumerate(tqdm(batches, total=n_steps, desc=f"seed {seed} ep {epoch}", leave=False), 1):
            x, y, w = (t.to(device, non_blocking=True) for t in batch)
            if bal is not None:
                xb, yb, _ = (t.to(device, non_blocking=True) for t in bal)
                x, y = balanced_mixup(x, y, xb, yb, args.mixup_alpha)
                w = None
            with torch.autocast(device.type, enabled=args.amp):
                logits = model(x).float()
                loss = criterion(logits, y, w)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(resize_to(x, args.teacher_size)).float()
                    loss = (1 - args.kd_beta) * loss + args.kd_beta * kd_loss(logits, t_logits, args.kd_T)
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0 or step == n_steps:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * len(y)
            seen += len(y)
        train_loss = total / max(seen, 1)

        yv, pv = predict(model, loaders["val"], device, args.amp)
        m = compute_metrics(yv, pv)
        sched.step(m["prod"])
        improved = m["prod"] > best_prod and (not args.require_loss_drop or train_loss < best_loss)
        if improved:
            best_prod, best_epoch = m["prod"], epoch
            torch.save(model.state_dict(), ckpt)
        best_loss = min(best_loss, train_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0,
               **{f"val_{k}": v for k, v in m.items()}, "saved": improved}
        log.append(row)
        print(f"seed {seed} ep {epoch:3d} loss {train_loss:.4f} val sens {m['sens']:.3f} spec {m['spec']:.3f} "
              f"auroc {m['auroc']:.3f}{' *' if improved else ''}", flush=True)
    pd.DataFrame(log).to_csv(out_dir / f"seed{seed}_log.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    result = {"seed": seed, "best_epoch": best_epoch, "params_M": count_params(model), "n_train": len(tr),
              "n_train_pos": n_pos}
    for k in eval_names:
        y, p = predict(model, loaders[k], device, args.amp)
        result.update({f"{k}_{mk}": mv for mk, mv in compute_metrics(y, p, prevalence=args.prevalence).items()})
        pd.DataFrame({"id": splits[k]["id"], "label": y, "prob": p}).to_csv(out_dir / f"seed{seed}_{k}_preds.csv",
                                                                           index=False)
    if not args.keep_ckpt:
        ckpt.unlink()
    return result


def summarise(results: pd.DataFrame, splits):
    summary = {}
    for split in splits:
        if split == "train":
            continue
        for k in ["sens", "spec", "gm", "npv", "npv_adj", "auroc"]:
            col = results[f"{split}_{k}"] * 100
            summary[f"{split}_{k}"] = f"{col.mean():.2f} ± {col.std(ddof=0):.2f}"
    best = results.loc[results["test_gm"].idxmax()]
    summary["best_pair"] = f"({best['test_sens'] * 100:.2f}, {best['test_spec'] * 100:.2f}) seed {int(best['seed'])}"
    summary["best_npv_adj"] = f"{best['test_npv_adj'] * 100:.2f}"
    return summary


def default_out(args):
    parts = [args.model, args.aug, args.loss]
    if args.sampling != "none":
        parts.append(args.sampling)
    if args.mixup_alpha:
        parts.append(f"mixup{args.mixup_alpha}")
    if args.iqa != "none":
        parts.append(args.iqa)
    if args.teacher:
        parts.append(f"kd-{args.teacher}-b{args.kd_beta}")
    if args.fold is not None:
        parts.append(f"fold{args.fold}")
    if getattr(args, "lr", 1e-3) != 1e-3:
        parts.append(f"lr{args.lr:g}")
    return Path("results") / args.manifest.stem / "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="proc_path in the manifest is relative to this")
    ap.add_argument("--model", default="mobilevitv2")
    ap.add_argument("--aug", default="none", choices=["none", "geometric", "zoom", "colour", "sam"])
    ap.add_argument("--sam-prob", type=float, default=0.5, help="probability of masking a training image (--aug sam)")
    ap.add_argument("--loss", default="bce", choices=["bce", "wce", "focal"])
    ap.add_argument("--sampling", default="none", choices=["none", "under", "over"])
    ap.add_argument("--mixup-alpha", type=float, default=0.0, help="Balanced Mix-up alpha (0 = off)")
    ap.add_argument("--iqa", default="none", choices=["none", "niqe", "clipiqa"])
    ap.add_argument("--iqa-scores", type=Path, default=Path("data/iqa_scores.csv"))
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--teacher", default=None, help="teacher model key for knowledge distillation")
    ap.add_argument("--teacher-ckpt", type=Path, default=None)
    ap.add_argument("--kd-beta", type=float, default=0.5, help="weight of the KD term (1.0 = distillation only)")
    ap.add_argument("--kd-T", type=float, default=4.0)
    ap.add_argument("--prevalence", type=float, default=4887 / 29574,
                    help="suspicious prevalence for adjusted NPV/PPV (default: the paper's dataset, 16.5%%)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (for small GPUs)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--input-size", type=int, default=None, help="override the model's default input size")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--scratch", action="store_true", help="no ImageNet weights")
    ap.add_argument("--require-loss-drop", action="store_true",
                    help="official-code rule: save only if val sens*spec AND train loss both improve")
    ap.add_argument("--keep-ckpt", action="store_true")
    ap.add_argument("--max-train", type=int, default=None, help="debug: subsample the training split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    splits = get_splits(args)
    out_dir = args.out or default_out(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(args).items()}, open(out_dir / "config.json", "w"), indent=2)
    for k, df in splits.items():
        print(f"{k:9s} n={len(df):5d}  suspicious={int(df['label'].sum()):5d}")

    teacher = None
    if args.teacher:
        teacher, args.teacher_size = load_teacher(args, device)

    results = []
    for seed in args.seeds:
        results.append(run_seed(args, seed, splits, device, out_dir, teacher))
        pd.DataFrame(results).to_csv(out_dir / "seeds.csv", index=False)
    summary = summarise(pd.DataFrame(results), splits)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
