"""Person 2: print the model comparison from results/experiments.csv (for the viva / report).

  python -m src.models.show_results

Prints 4 small tables, saves results/comparison_table.csv, results/comparison_table.md
and (if matplotlib is installed) results/figures/model_comparison.png.
Only reads the log. Nothing is re-trained, so it runs in a second.
If a model was run more than once, the LAST full run (all CV folds) is used.
"""
from pathlib import Path

import pandas as pd

from src.common.config import load_config


def load_log(path):
    d = pd.read_csv(path)
    d = d[d["cv_folds"] == d["cv_folds"].max()]          # drop smoke-test runs (fewer folds)
    return d.drop_duplicates("model", keep="last").reset_index(drop=True)


def pm(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def build_tables(d):
    t1 = pd.DataFrame({
        "model": d["model"],
        "CV Top-1": [pm(a, b) for a, b in zip(d.cv_top1_all_mean, d.cv_top1_all_std)],
        "Student Top-1": d.st_top1_all.round(3),
        "Student Top-1 (true road was a candidate)": d.st_top1_in_cands.round(3),
    })
    t2 = pd.DataFrame({
        "model": d["model"],
        "CV macro F1": [pm(a, b) for a, b in zip(d.cv_macro_f1_mean, d.cv_macro_f1_std)],
        "Student macro F1": d.st_macro_f1.round(3),
        "Precision": d.st_precision.round(3),
        "Recall": d.st_recall.round(3),
        "Log loss (student)": d.st_log_loss.round(4),
    })
    gap = d.cv_train_top1_all_mean - d.cv_top1_all_mean
    t3 = pd.DataFrame({
        "model": d["model"],
        "Train Top-1": d.cv_train_top1_all_mean.round(3),
        "CV (validation) Top-1": d.cv_top1_all_mean.round(3),
        "Gap (train - val)": gap.round(3),
        "Reading": ["n/a (not measured)" if pd.isna(g) else
                    "overfits" if g > 0.05 else "slight gap" if g > 0.02 else "no real overfit"
                    for g in gap],
    })
    t4 = pd.DataFrame({
        "model": d["model"],
        "Route continuity (1-hop)": d.st_route_continuity_1hop.round(3),
        "Train time (s)": d.train_time_s.round(1),
        "Inference per trip (s)": d.infer_s_per_trip.round(4),
    })
    return {"1. Accuracy": t1, "2. F1, precision, recall, log loss": t2,
            "3. Overfitting check": t3, "4. Route continuity and cost": t4}


def make_chart(d, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped the chart. pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = range(len(d))
    ax[0].bar(x, d.cv_top1_all_mean, yerr=d.cv_top1_all_std, capsize=4)
    ax[0].set_title("CV Top-1 (higher is better)")
    ax[1].bar(x, d.st_top1_all)
    ax[1].set_title("Student-test Top-1 (higher is better)")
    ax[2].bar(x, d.st_log_loss)
    ax[2].set_title("Student-test log loss (lower is better)")
    for a in ax:
        a.set_xticks(list(x))
        a.set_xticklabels(d["model"], rotation=30)
    ax[0].set_ylim(0.8, 1.0)
    ax[1].set_ylim(0.8, 1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"chart saved: {out}")


def main():
    results = load_config()["paths"]["results"]
    d = load_log(Path(results) / "experiments.csv")
    tables = build_tables(d)
    md = []
    for name, t in tables.items():
        print(f"\n=== {name} ===")
        print(t.to_string(index=False))
        md.append(f"### {name}\n\n{t.to_markdown(index=False)}\n" if _has_tabulate() else
                  f"### {name}\n\n```\n{t.to_string(index=False)}\n```\n")
    pd.concat([t.set_index("model") for t in tables.values()], axis=1).to_csv(Path(results) / "comparison_table.csv")
    (Path(results) / "comparison_table.md").write_text("\n".join(md))
    make_chart(d, Path(results) / "figures" / "model_comparison.png")
    print("\nReference: nearest-road baseline about 54%; ceiling about 99% (true road must be a candidate).")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
