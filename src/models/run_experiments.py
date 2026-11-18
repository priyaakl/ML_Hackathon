"""Person 2: train + compare models. One command per model (or several).

Examples (run from the repo root):
  python -m src.models.run_experiments --models lr --sample-trips 10 --folds 3     # quick smoke test
  python -m src.models.run_experiments --models lr lgbm                            # real run
  python -m src.models.run_experiments --models lgbm --params models/best_params.json --tag lgbm_tuned

For each model it does:
  1. group-aware CV on the train trips (valid labels only)        -> CV mean/std of every metric + train/val gap
  2. fit on ALL train trips, score the student_test trips          -> student-test metrics, train/inference time
  3. append one row to results/experiments.csv, per-fold rows to results/cv_folds/<tag>.csv
student_test is used ONLY for comparing models, never for fitting or tuning.
"""
import argparse
import getpass
import json
import time
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.models import harness as H
from src.models.model_zoo import FAMILY, LOSS, build_model, config_of


def run_one(name, tag, params, args, train, n_train, st, n_st, feats, cfg, results):
    knn_gap = 0 if name == "knn" else args.gap_trips   # KNN scores its own training points as perfect: meaningless
    cv = H.cross_validate(lambda: build_model(name, args.seed, params), train, n_train, feats,
                          n_splits=args.folds, gap_trips=knn_gap, seed=args.seed, desc=f"CV {tag}")
    (results / "cv_folds").mkdir(parents=True, exist_ok=True)
    cv.to_csv(results / "cv_folds" / f"{tag}.csv", index=False)

    # ---- final fit on all training trips, then student_test
    model = build_model(name, args.seed, params)
    t0 = time.perf_counter()
    if name == "lgbm":   # eval_set is for LOGGING the loss curve only; nothing is tuned or stopped on it
        model.fit(train[feats], train["label"], eval_set=[(train[feats], train["label"]), (st[feats], st["label"])],
                  eval_names=["train", "student_test"], eval_metric="binary_logloss")
        ev = model.evals_result_
        curves = pd.DataFrame({"train_logloss": ev["train"]["binary_logloss"],
                               "student_test_logloss": ev["student_test"]["binary_logloss"]})
        (results / "curves").mkdir(parents=True, exist_ok=True)
        curves.to_csv(results / "curves" / f"{tag}_loss_curve.csv", index_label="iteration")
    else:
        model.fit(train[feats], train["label"])
    train_s = time.perf_counter() - t0
    if name == "mlp":
        (results / "curves").mkdir(parents=True, exist_ok=True)
        pd.Series(model[-1].loss_curve_, name="train_loss").to_csv(results / "curves" / f"{tag}_loss_curve.csv",
                                                                   index_label="epoch")

    t0 = time.perf_counter()
    p = model.predict_proba(st[feats])[:, 1]
    infer_s = time.perf_counter() - t0
    m = H.evaluate(st, p, n_st)
    cont = H.route_continuity_1hop(st, p, cfg["paths"]["data_root"])

    n_trips_st = st["trip_id"].nunique()
    mean, std = cv.mean(numeric_only=True), cv.std(numeric_only=True)
    row = dict(
        model=tag, family=FAMILY[name], config=json.dumps(config_of(name, params), default=str),
        loss=LOSS[name], seed=args.seed, cv_folds=args.folds, n_train_rows=len(train),
        cv_top1_all_mean=mean["top1_all"], cv_top1_all_std=std["top1_all"],
        cv_macro_f1_mean=mean["macro_f1"], cv_macro_f1_std=std["macro_f1"],
        cv_log_loss_mean=mean["log_loss"], cv_log_loss_std=std["log_loss"],
        cv_train_top1_all_mean=mean.get("train_top1_all", float("nan")),
        cv_train_log_loss_mean=mean.get("train_log_loss", float("nan")),
        st_top1_all=m["top1_all"], st_top1_in_cands=m["top1_in_cands"], st_miss_rate=m["miss_rate"],
        st_macro_f1=m["macro_f1"], st_precision=m["precision"], st_recall=m["recall"], st_log_loss=m["log_loss"],
        st_route_continuity_1hop=cont,
        train_time_s=train_s, cv_fit_time_mean_s=mean["fit_s"],
        infer_time_s_student_test=infer_s, infer_ms_per_1000_rows=1000 * infer_s / len(st) * 1000,
        infer_s_per_trip=infer_s / n_trips_st, who=args.who, notes=args.notes,
    )
    H.log_experiment(row, results)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lr"], help="lr rf lgbm knn mlp gnb")
    ap.add_argument("--tag", default=None, help="name in the log (only when running one model)")
    ap.add_argument("--params", default=None, help="JSON file with hyperparameter overrides (one model)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gap-trips", type=int, default=10, help="training trips scored per fold for the overfit gap")
    ap.add_argument("--sample-trips", type=int, default=None, help="use only N trips per partition (smoke test)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--who", default=getpass.getuser())
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_config()
    args.seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    results = Path(args.results) if args.results else cfg["paths"]["results"]
    params = json.load(open(args.params)) if args.params else None
    if (args.tag or args.params) and len(args.models) != 1:
        raise SystemExit("--tag / --params only work with exactly one model")

    feats = H.load_feature_columns()
    print("loading data ...")
    train, n_train = H.load_partition("train", args.processed, args.sample_trips, args.seed)
    st, n_st_by_trip = H.load_partition("student_test", args.processed, args.sample_trips, args.seed)
    n_st = int(n_st_by_trip.sum())
    print(f"train: {len(train):,} rows, {train.trip_id.nunique()} trips | "
          f"student_test: {len(st):,} rows, {st.trip_id.nunique()} trips")

    out = []
    for name in args.models:
        tag = args.tag or name
        out.append(run_one(name, tag, params, args, train, n_train, st, n_st, feats, cfg, results))
    cols = ["model", "cv_top1_all_mean", "cv_top1_all_std", "cv_macro_f1_mean", "st_top1_all", "st_macro_f1",
            "st_log_loss", "st_route_continuity_1hop", "train_time_s", "infer_s_per_trip"]
    print("\n" + pd.DataFrame(out)[cols].round(4).to_string(index=False))
    print(f"\nlogged to {results / 'experiments.csv'}")


if __name__ == "__main__":
    main()
