"""Demo smoke test: run it from the UI laptop (Laptop 2) BEFORE the evaluator arrives.

It plays the role of the UI. Over HTTP, exactly like the UI, it:
  1. calls GET /health   -> checks the API is reachable and that NO mock scorer/decoder is active
  2. calls GET /trips    -> lists the trips the evaluator can pick
  3. calls POST /predict -> for every trip (or the ones you name), and checks each response

Checks per trip (any FAIL makes the script exit with code 1):
  * HTTP 200 and a reasonable response time
  * one predicted arc per GPS point, for both the HMM route and the classifier-only route
  * the response is not labelled MOCK
  * the response exposes only the agreed fields, and `metrics` holds summary numbers only
    (no per-point arrays), so the reference route never leaves the server
Trips whose folder has no .route file get no metrics: that is expected for hidden trips and is a WARN, not a FAIL.

Uses only the Python standard library, so Laptop 2 needs nothing installed.

Usage (from the repo root):
    python -m src.api.demo_check --api http://<LAPTOP-1-IP>:8000
    python -m src.api.demo_check                       # uses PATHSNAP_API_URL, else config.yaml ui.api_base_url
    python -m src.api.demo_check --trips 00000009 00000065
Writes results/demo_check.csv (a record you can screenshot or attach to the report).
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_KEYS = {
    "trip_id", "mode", "n_observations", "n_candidate_rows", "n_obs_without_candidates", "predicted_arcs",
    "classifier_only_arcs", "gps", "route_geometry", "route_segments", "timing", "metrics", "request_s",
}
SLOW_S = 30.0          # warn above this per-trip request time


def default_api() -> str:
    if os.environ.get("PATHSNAP_API_URL"):
        return os.environ["PATHSNAP_API_URL"]
    try:
        import yaml
        with open(ROOT / "config.yaml") as f:
            return yaml.safe_load(f)["ui"]["api_base_url"]
    except Exception:
        return "http://127.0.0.1:8000"


def call(base: str, path: str, body: dict = None, timeout: float = 300.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = {}
        return e.code, detail, time.perf_counter() - t0


def check_trip(res: dict) -> tuple:
    """Return (fails, warns) for one /predict response."""
    fails, warns = [], []
    n = res.get("n_observations")
    if len(res.get("predicted_arcs", [])) != n:
        fails.append(f"predicted_arcs has {len(res.get('predicted_arcs', []))} entries, expected {n}")
    if len(res.get("classifier_only_arcs", [])) != n:
        fails.append("classifier_only_arcs length differs from the number of GPS points")
    if len(res.get("gps", [])) != n:
        fails.append("gps length differs from the number of GPS points")
    if res.get("mode", {}).get("is_mock"):
        fails.append("response is labelled MOCK (config.yaml use_mock_scorer/use_mock_decoder is true)")
    extra = set(res) - EXPECTED_KEYS
    if extra:
        fails.append(f"unexpected fields in response (possible leak): {sorted(extra)}")
    m = res.get("metrics") or {}
    if any(isinstance(v, list) for v in m.values()):
        fails.append("metrics contains a list: per-point data must not leave the server")
    if not m.get("available"):
        warns.append("no metrics: " + str(m.get("reason", "metrics missing")))
    if res.get("n_obs_without_candidates", 0) == n and n:
        warns.append("no GPS point has a candidate road")
    return fails, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=default_api(), help="API base URL, e.g. http://192.168.1.5:8000")
    ap.add_argument("--trips", nargs="*", help="trip ids to test (default: every trip the API lists)")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    api = args.api.rstrip("/")
    print(f"API under test: {api}\n")
    problems = 0

    # 1) health
    try:
        code, h, dt = call(api, "/health", timeout=10)
    except Exception as e:
        print(f"FAIL  cannot reach {api}/health: {e}\n"
              "      Check: backend running on Laptop 1, same network, firewall allows Python, IP is current.")
        return 1
    mode = h.get("mode", {})
    print(f"/health  -> {code} in {dt:.2f}s | status={h.get('status')} | trips={h.get('n_trips')}")
    print(f"           scorer : {mode.get('scorer')}")
    print(f"           decoder: {mode.get('decoder')} ({mode.get('decoder_params')})")
    if code != 200 or h.get("status") != "ok":
        print("FAIL  /health is not ok")
        return 1
    if mode.get("is_mock"):
        print("FAIL  MOCK components are active. Set use_mock_scorer: false and use_mock_decoder: false in config.yaml, restart the API.")
        problems += 1

    # 2) trips
    code, t, dt = call(api, "/trips", timeout=30)
    if code != 200:
        print(f"FAIL  /trips returned {code}: {t.get('detail')}")
        return 1
    trips = args.trips or t.get("trips", [])
    print(f"/trips   -> {len(t.get('trips', []))} trips available; testing {len(trips)}\n")

    # 3) predict every trip
    rows = []
    print(f"{'trip':<10}{'points':>8}{'time s':>8}{'HMM top1':>10}{'clf top1':>10}{'HMM cont.':>11}  result")
    for tid in trips:
        try:
            code, res, dt = call(api, "/predict", {"trip_id": tid}, timeout=args.timeout)
        except Exception as e:
            print(f"{tid:<10}{'':>8}{'':>8}{'':>10}{'':>10}{'':>11}  FAIL request error: {e}")
            problems += 1
            rows.append({"trip_id": tid, "status": "FAIL", "notes": f"request error: {e}"})
            continue
        if code != 200:
            print(f"{tid:<10}{'':>8}{dt:>8.1f}{'':>10}{'':>10}{'':>11}  FAIL HTTP {code}: {res.get('detail')}")
            problems += 1
            rows.append({"trip_id": tid, "status": "FAIL", "notes": f"HTTP {code}: {res.get('detail')}"})
            continue
        fails, warns = check_trip(res)
        if dt > SLOW_S:
            warns.append(f"slow: {dt:.1f}s")
        m = res.get("metrics") or {}
        hm, cl = (m.get("hmm") or {}), (m.get("classifier_only") or {})
        status = "FAIL" if fails else ("WARN" if warns else "OK")
        problems += bool(fails)
        f = lambda v: "-" if v is None else f"{v:.4f}"
        print(f"{tid:<10}{res.get('n_observations', 0):>8}{dt:>8.1f}{f(hm.get('top1_accuracy')):>10}"
              f"{f(cl.get('top1_accuracy')):>10}{f(hm.get('route_continuity')):>11}  {status} {'; '.join(fails + warns)}")
        rows.append({"trip_id": tid, "n_obs": res.get("n_observations"), "request_s": round(dt, 2),
                     "server_total_s": (res.get("timing") or {}).get("total_s"),
                     "hmm_top1": hm.get("top1_accuracy"), "clf_top1": cl.get("top1_accuracy"),
                     "hmm_macro_f1": hm.get("macro_f1"), "clf_macro_f1": cl.get("macro_f1"),
                     "hmm_continuity": hm.get("route_continuity"), "clf_continuity": cl.get("route_continuity"),
                     "status": status, "notes": "; ".join(fails + warns)})

    out = ROOT / "results" / "demo_check.csv"
    out.parent.mkdir(exist_ok=True)
    keys = ["trip_id", "n_obs", "request_s", "server_total_s", "hmm_top1", "clf_top1", "hmm_macro_f1",
            "clf_macro_f1", "hmm_continuity", "clf_continuity", "status", "notes"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "OK" for r in rows)
    print(f"\n{ok} OK, {sum(r['status'] == 'WARN' for r in rows)} WARN, {sum(r['status'] == 'FAIL' for r in rows)} FAIL "
          f"of {len(rows)} trips. Saved {out.relative_to(ROOT)}")
    print("Now check Laptop 1's terminal: it should show one log line per request from THIS laptop's IP.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
