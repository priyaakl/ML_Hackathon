"""Report figures and the 3.1 comparison table, built ONLY from files already in the repo.

  python -m src.models.report_figures

Reads : results/experiments.csv, results/curves/*.csv, results/candidate_stats.csv
Writes: results/figures/fig_cv_macro_f1.png, fig_student_top1_f1.png, fig_log_loss.png,
        fig_loss_curves.png, fig_overfit_gap.png, fig_cost.png
        results/table_3_1.md   (the professor's comparison table + an extended table)
Nothing is re-trained. If a model appears in experiments.csv more than once, the LAST full CV run is used
(same rule as show_results.py). A tuned run logged under its own tag (e.g. lgbm_tuned) appears as its own bar.
Re-run it after every new row in experiments.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.config import load_config

LABELS = {"lr": "Logistic\nRegression", "lgbm": "LightGBM", "rf": "Random\nForest", "mlp": "MLP",
          "gnb": "Gaussian\nNaive Bayes", "knn": "KNN", "lgbm_tuned": "LightGBM\n(tuned)"}
NAMES = {k: v.replace("\n", " ") for k, v in LABELS.items()}
GREY, BLUE, ORANGE, GREEN = "#7b8998", "#215cb5", "#e08a1e", "#147a58"


def load_log(path):
    d = pd.read_csv(path)
    d = d[d["cv_folds"] == d["cv_folds"].max()]
    return d.drop_duplicates("model", keep="last").reset_index(drop=True)


def colour(m):
    return ORANGE if m.endswith("_tuned") else BLUE


def bars(ax, d, col, err=None, ylim=None, fmt="{:.3f}", logy=False):
    x = np.arange(len(d))
    ax.bar(x, d[col], color=[colour(m) for m in d.model], yerr=None if err is None else d[err],
           capsize=4, edgecolor="none")
    for xi, v in zip(x, d[col]):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in d.model], fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path)


def main():
    cfg = load_config()
    res = cfg["paths"]["results"]
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    d = load_log(res / "experiments.csv")
    # reference lines for Top-1 on student test (from Person 1's candidate_stats.csv)
    base = ceil = None
    try:
        cs2 = pd.read_csv(res / "candidate_stats.csv")
        row = cs2[cs2["partition"] == "student_test"].iloc[0]
        base, ceil = row["nearest_arc_top1_pct"] / 100, 1 - row["candidate_miss_rate_pct"] / 100
    except Exception:
        pass

    # 1. CV macro F1 (mean +- std over the 5 group folds)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars(ax, d, "cv_macro_f1_mean", "cv_macro_f1_std", (0.8, 1.0))
    ax.set(ylabel="macro F1 (axis starts at 0.80)", title="5-fold group CV macro F1 (mean ± std over folds, 80 train trips)")
    save(fig, figs / "fig_cv_macro_f1.png")

    # 2. student-test Top-1 and macro F1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(ax[0], d, "st_top1_all", None, (0.5, 1.02))
    if base:
        ax[0].axhline(base, color=GREY, ls="--", lw=1)
        ax[0].text(len(d) - 0.55, base + 0.012, f"nearest-arc baseline {base:.3f}", ha="right", fontsize=8, color="#3b4654",
                   zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
        ax[0].axhline(ceil, color=GREEN, ls="--", lw=1)
        ax[0].text(len(d) - 0.55, ceil - 0.035, f"ceiling {ceil:.4f} (candidate coverage)", ha="right", fontsize=8, color=GREEN,
                   zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    ax[0].set(ylabel="Top-1 (axis starts at 0.50)", title="Student-test Top-1 (per GPS point)")
    bars(ax[1], d, "st_macro_f1", None, (0.8, 1.0))
    ax[1].set(ylabel="macro F1 (axis starts at 0.80)", title="Student-test macro F1 (per candidate)")
    save(fig, figs / "fig_student_top1_f1.png")

    # 3. log loss (CV and student test), log scale because Naive Bayes is far worse
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(ax[0], d, "cv_log_loss_mean", "cv_log_loss_std", fmt="{:.3f}", logy=True)
    ax[0].set(ylabel="log loss (log scale)", title="CV log loss (mean ± std)")
    bars(ax[1], d, "st_log_loss", None, fmt="{:.3f}", logy=True)
    ax[1].set(ylabel="log loss (log scale)", title="Student-test log loss")
    save(fig, figs / "fig_log_loss.png")

    # 4. real loss histories only (LightGBM, MLP)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lg = res / "curves" / "lgbm_loss_curve.csv"
    if lg.exists():
        c = pd.read_csv(lg)
        ax[0].plot(c.iteration + 1, c.train_logloss, label="train", color=BLUE)
        ax[0].plot(c.iteration + 1, c.student_test_logloss, label="student test (monitored only)", color=ORANGE)
        ax[0].set(xlabel="boosting iteration", ylabel="binary log loss",
                  title="LightGBM loss history (LightGBM's own metric;\nthe tables clip probabilities at 1e-7, so final values differ slightly)")
        ax[0].legend()
    ml = res / "curves" / "mlp_loss_curve.csv"
    if ml.exists():
        c = pd.read_csv(ml)
        ax[1].plot(c.epoch + 1, c.train_loss, color=BLUE, label="training loss")
        ax[1].set(xlabel="epoch", ylabel="loss (BCE + L2)", title=f"MLP training loss ({len(c)} epochs, no validation curve logged)")
        ax[1].legend()
    save(fig, figs / "fig_loss_curves.png")

    # 5. overfitting check: train vs CV Top-1
    g = d.dropna(subset=["cv_train_top1_all_mean"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.cv_train_top1_all_mean, 0.4, label="train (in-fold)", color=GREY)
    ax.bar(x + 0.2, g.cv_top1_all_mean, 0.4, label="validation (held-out trips)", color=BLUE)
    for xi, a, b in zip(x, g.cv_train_top1_all_mean, g.cv_top1_all_mean):
        ax.text(xi, max(a, b) + 0.003, f"gap {a - b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in g.model], fontsize=8)
    ax.set(ylim=(0.8, 0.96), ylabel="Top-1 (axis starts at 0.80)", title="Train vs validation Top-1 (KNN omitted: train score not measured)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, figs / "fig_overfit_gap.png")

    # 6. cost: training time and inference time per trip (log scale)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(ax[0], d, "train_time_s", fmt="{:.1f}", logy=True)
    ax[0].set(ylabel="seconds (log scale)", title="Training time (full train set; KNN on a 250k sample)")
    bars(ax[1], d, "infer_s_per_trip", fmt="{:.4f}", logy=True)
    ax[1].set(ylabel="seconds per trip (log scale)", title="Inference time per student-test trip")
    save(fig, figs / "fig_cost.png")

    # ---- table 3.1
    t = pd.DataFrame({
        "Model": [NAMES.get(m, m) for m in d.model],
        "Algorithm Family": d.family,
        "Key Configuration": [_cfg(c) for c in d.config],
        "CV Macro F1-score": [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.cv_macro_f1_mean, d.cv_macro_f1_std)],
        "Top-1 Accuracy": [f"{v:.3f}" for v in d.st_top1_all],
        "Route continuity": [f"{v:.3f}" for v in d.st_route_continuity_1hop],
        "Student-Test Macro F1-score": [f"{v:.3f}" for v in d.st_macro_f1],
        "Log Loss": [f"{v:.4f}" for v in d.st_log_loss],
        "Inference Time (s per trip)": [f"{v:.4f}" for v in d.infer_s_per_trip],
    })
    ext = pd.DataFrame({
        "Model": [NAMES.get(m, m) for m in d.model],
        "CV Top-1": [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.cv_top1_all_mean, d.cv_top1_all_std)],
        "CV log loss": [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.cv_log_loss_mean, d.cv_log_loss_std)],
        "Student precision": [f"{v:.3f}" for v in d.st_precision], "Student recall": [f"{v:.3f}" for v in d.st_recall],
        "Train Top-1": ["" if pd.isna(v) else f"{v:.3f}" for v in d.cv_train_top1_all_mean],
        "Train time (s)": [f"{v:.1f}" for v in d.train_time_s], "Seed": d.seed, "CV folds": d.cv_folds,
    })
    note = ("Partitions: CV columns = 5-fold GroupKFold over the 80 train trips (valid-label rows only). "
            "Top-1 Accuracy, Route continuity, Student-Test Macro F1, Log Loss and Inference Time = the 10 student_test trips "
            "(never used for fitting). Top-1 = per GPS point, points whose true arc is not a candidate count as wrong. "
            "Macro F1 = candidate-level, threshold 0.5. Route continuity = share of consecutive predicted arcs that are the "
            "same or directly connected (classifier only, no HMM).")
    (res / "table_3_1.md").write_text(
        "### 3.1 Overall model comparison\n\n" + md(t) + "\n\n" + note +
        "\n\n#### Extended columns (precision, recall, train score, cost)\n\n" + md(ext) + "\n")
    print("  saved", res / "table_3_1.md")


def md(df):
    """Plain markdown table (no tabulate dependency)."""
    rows = [list(map(str, df.columns))] + [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _cfg(c):
    import json
    p = json.loads(c)
    p = {k: v for k, v in p.items() if k not in ("n_jobs", "verbose")}
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "library defaults"


if __name__ == "__main__":
    main()
