"""Person 2: fit ONE model on all valid training trips and save it for the API.

  python -m src.models.train_final --model lr                                  # Milestone B baseline
  python -m src.models.train_final --model lgbm --params models/best_params.json

Writes models/final_model.joblib and models/final_model_meta.json, then reloads it through
CandidateScorer and prints student_test Top-1 as a sanity check (student_test is not used for fitting).
"""
import argparse
import datetime
import json
from pathlib import Path

import joblib

from src.common.config import ROOT, load_config
from src.models import harness as H
from src.models.model_zoo import build_model, config_of
from src.models.scorer import CandidateScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--processed", default=None)
    ap.add_argument("--sample-trips", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg.get("seed", 42)
    params = json.load(open(args.params)) if args.params else None
    feats = H.load_feature_columns()
    train, _ = H.load_partition("train", args.processed, args.sample_trips, seed)

    model = build_model(args.model, seed, params)
    print(f"fitting {args.model} on {len(train):,} rows ...")
    model.fit(train[feats], train["label"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "final_model.joblib", compress=3)
    with open(out / "final_model_meta.json", "w") as f:
        json.dump(dict(model=args.model, config=config_of(args.model, params), seed=seed,
                       n_train_rows=len(train), trained=datetime.datetime.now().isoformat(timespec="seconds"),
                       feature_columns=feats), f, indent=2, default=str)

    st, n_st = H.load_partition("student_test", args.processed, args.sample_trips, seed)
    scorer = CandidateScorer(str(out / "final_model.joblib"), str(ROOT / "models" / "feature_columns.json"))
    m = H.evaluate(st, scorer.predict_proba(st), int(n_st.sum()))
    print(f"saved. reload check on student_test: top1_all={m['top1_all']:.4f} log_loss={m['log_loss']:.4f}")


if __name__ == "__main__":
    main()
