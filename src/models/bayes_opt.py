"""Person 2: Bayesian hyperparameter search (Optuna TPE) for LightGBM, using group-aware CV on TRAIN trips only.

  python -m src.models.bayes_opt --n-trials 5 --sample-trips 10 --folds 3     # smoke test
  python -m src.models.bayes_opt --n-trials 30                                # real run (can be stopped and resumed)

Keeps: results/optuna.db (the study, resumable), results/optuna_trials.csv (every trial),
       results/optuna_summary.json, models/best_params.json.
Then compare tuned vs untuned with:
  python -m src.models.run_experiments --models lgbm --params models/best_params.json --tag lgbm_tuned
"""
import argparse
import json
from pathlib import Path

import optuna

from src.common.config import ROOT, load_config
from src.models import harness as H
from src.models.model_zoo import build_model


def suggest(trial):
    """Search space (report question 2). Ranges are deliberately moderate: wide enough to matter,
    narrow enough to find something good in ~30 trials."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30, help="NEW trials to run in this call")
    ap.add_argument("--objective", choices=["top1", "logloss"], default="top1",
                    help="top1 = maximise CV observation-level Top-1; logloss = minimise CV log loss")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample-trips", type=int, default=None)
    ap.add_argument("--timeout-min", type=float, default=None, help="stop after this many minutes")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg.get("seed", 42)
    results = Path(args.results) if args.results else cfg["paths"]["results"]
    results.mkdir(parents=True, exist_ok=True)
    feats = H.load_feature_columns()
    train, n_train = H.load_partition("train", args.processed, args.sample_trips, seed)   # NO student_test here

    def objective(trial):
        params = suggest(trial)
        cv = H.cross_validate(lambda: build_model("lgbm", seed, params), train, n_train, feats,
                              n_splits=args.folds, gap_trips=0, seed=seed, desc=f"trial {trial.number}")
        for k in ("top1_all", "macro_f1", "log_loss", "fit_s"):
            trial.set_user_attr(f"cv_{k}", float(cv[k].mean()))
        return float(cv["top1_all"].mean()) if args.objective == "top1" else float(cv["log_loss"].mean())

    study = optuna.create_study(
        study_name=f"pathsnap_lgbm_{args.objective}", storage=f"sqlite:///{results / 'optuna.db'}",
        load_if_exists=True, direction="maximize" if args.objective == "top1" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed))

    def save(study, _trial):
        study.trials_dataframe().to_csv(results / "optuna_trials.csv", index=False)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_min and args.timeout_min * 60,
                   callbacks=[save], show_progress_bar=True)

    save(study, None)
    done = [t for t in study.trials if t.duration is not None]
    summary = dict(objective=args.objective, n_trials=len(study.trials), cv_folds=args.folds,
                   best_value=study.best_value, best_params=study.best_params,
                   total_compute_min=round(sum(t.duration.total_seconds() for t in done) / 60, 1))
    json.dump(summary, open(results / "optuna_summary.json", "w"), indent=2)
    (ROOT / "models").mkdir(exist_ok=True)
    json.dump(study.best_params, open(ROOT / "models" / "best_params.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
