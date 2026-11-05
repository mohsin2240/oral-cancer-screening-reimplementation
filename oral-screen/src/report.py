"""Aggregate finished runs into the paper's tables (IV-IX) as CSV + Markdown under results/tables/.

Each cell is mean +- std over seeds (test split), plus the "best pair" (sens, spec) of the highest-test-GM seed and GM
of the means, matching the paper's layout. Missing runs are left blank.

Usage: python src/report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import STUDENTS, TEACHERS
from run_grid import teacher_dir

ROOT = Path("results/manifest_train5")
OUT = Path("results/tables")
sys.stdout.reconfigure(encoding="utf-8")  # the Windows console codepage can't print β / ±


def stats(run: Path, split="test"):
    f = run / "seeds.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    b = s.loc[s[f"{split}_gm"].idxmax()]
    sens, spec = s[f"{split}_sens"] * 100, s[f"{split}_spec"] * 100
    return {
        "Sensitivity": f"{sens.mean():.2f}±{sens.std(ddof=0):.2f}",
        "Specificity": f"{spec.mean():.2f}±{spec.std(ddof=0):.2f}",
        "Best pair": f"({b[f'{split}_sens'] * 100:.2f},{b[f'{split}_spec'] * 100:.2f})",
        "GM": f"{np.sqrt(sens.mean() * spec.mean()):.2f}",
        "AUROC": f"{s[f'{split}_auroc'].mean() * 100:.2f}±{s[f'{split}_auroc'].std(ddof=0) * 100:.2f}",
        "NPV@16.5%": f"{s[f'{split}_npv_adj'].mean() * 100:.2f}" if f"{split}_npv_adj" in s else "",
        "seeds": len(s),
    }


def model_table(rows: dict, name: str):
    """rows: {method label: config suffix}; one column block per student model."""
    recs = []
    for label, suffix in rows.items():
        for m in STUDENTS:
            st = stats(ROOT / f"{m}_{suffix}")
            if st:
                recs.append({"Method": label, "Model": m, **st})
    if not recs:
        return
    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"{name}.csv", index=False)
    wide = df.pivot(index="Method", columns="Model", values=["Sensitivity", "Specificity", "Best pair", "GM"])
    wide = wide.reindex(index=list(rows)).stack(0, future_stack=True).reindex(columns=[m for m in STUDENTS if m in df.Model.unique()])
    (OUT / f"{name}.md").write_text(wide.to_markdown(), encoding="utf-8")
    print(f"\n## {name}\n{wide.to_string()}")


def simple_table(rows: dict, name: str, split="test"):
    recs = [{"Config": k, **st} for k, run in rows.items() if (st := stats(run, split))]
    if not recs:
        return
    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"\n## {name}\n{df.to_string(index=False)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_table({"Baseline": "none_bce", "SAM output": "sam_bce", "Zoom-in": "zoom_bce",
                 "Geometric": "geometric_bce", "Colour jitter": "colour_bce"}, "table4_augmentation")
    model_table({"Baseline": "none_bce", "NIQE reweighting": "none_bce_niqe",
                 "CLIP-IQA reweighting": "none_bce_clipiqa"}, "table5_iqa")
    model_table({"Baseline": "none_bce", "Random undersampling": "none_bce_under",
                 "Random oversampling": "none_bce_over", "Weighted cross-entropy": "none_wce",
                 "Focal loss": "none_focal", "Balanced mix-up (0.1)": "none_bce_mixup0.1",
                 "Balanced mix-up (0.2)": "none_bce_mixup0.2", "Balanced mix-up (0.3)": "none_bce_mixup0.3"},
                "table6_imbalance")
    simple_table({f"Fold {f + 1}": ROOT / f"mobilevitv2_geometric_wce_fold{f}" for f in range(5)}, "table7_cv")
    rows = {"MobileViTv2 recipe (no KD)": ROOT / "mobilevitv2_geometric_wce"}
    rows |= {f"teacher {t}": teacher_dir(t) for t in TEACHERS}
    rows |= {f"KD {t} β={b}": ROOT / f"mobilevitv2_geometric_wce_kd-{t}-b{b}" for t in TEACHERS for b in (0.5, 1.0)}
    simple_table(rows, "table8_distillation")
    ext = Path("results/manifest_external")
    ext_runs = {"baseline": ext / "mobilevitv2_none_bce", "recipe": ext / "mobilevitv2_geometric_wce"}
    simple_table(ext_runs, "external_source_heldin")  # Mendeley test split
    simple_table(ext_runs, "external_source", split="external")  # the held-out Kaggle source
    for run in [ROOT / "mobilevitv2_geometric_wce"]:
        f = run / "robustness.csv"
        if f.exists():
            df = pd.read_csv(f, header=[0, 1], index_col=[0, 1]).round(1)
            df.to_csv(OUT / "table9_robustness.csv")
            (OUT / "table9_robustness.md").write_text(df.to_markdown(), encoding="utf-8")
            print(f"\n## table9_robustness\n{df.to_string()}")


if __name__ == "__main__":
    main()
