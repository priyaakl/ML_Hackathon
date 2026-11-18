"""PathSnap evaluator UI (Person 4). Runs on Laptop 2.

The UI does no map-matching itself. It only talks to the FastAPI backend on Laptop 1 over HTTP:
    GET  /health   -> is Laptop 1 reachable? which scorer/decoder is loaded?
    GET  /trips    -> trip ids for the dropdown
    POST /predict  -> predicted route, timing and summary metrics for the chosen trip

API address: config.yaml -> ui.api_base_url, overridden by the env var PATHSNAP_API_URL,
and editable in the sidebar.

Run from the repo root:   streamlit run src/ui/app.py
"""
import os
from pathlib import Path

import folium
import requests
import streamlit as st
import yaml
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[2]
GPS_COLOR, ROUTE_COLOR = "#2563eb", "#dc2626"
MAX_GPS_MARKERS = 1500          # draw at most this many GPS dots (long trips have 16k+ points)


# ----------------------------------------------------------------------------------------------
# Talking to the API
# ----------------------------------------------------------------------------------------------
def default_api_url() -> str:
    env = os.environ.get("PATHSNAP_API_URL")
    if env:
        return env
    try:
        with open(ROOT / "config.yaml") as f:
            return yaml.safe_load(f)["ui"]["api_base_url"]
    except Exception:
        return "http://127.0.0.1:8000"


def api_get(base: str, path: str, timeout: float = 5):
    r = requests.get(base.rstrip("/") + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_predict(base: str, trip_id: str) -> dict:
    r = requests.post(base.rstrip("/") + "/predict", json={"trip_id": trip_id}, timeout=300)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API returned {r.status_code}: {detail}")
    return r.json()


# ----------------------------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------------------------
def build_map(result: dict) -> folium.Map:
    gps = result["gps"]                                 # [[lat, lon], ...]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(30, 30))

    # Predicted route: one line per arc, so gaps are never drawn as invented straight roads.
    route = folium.FeatureGroup(name="Predicted route").add_to(m)
    for seg in result["route_segments"]:
        folium.PolyLine(seg, color=ROUTE_COLOR, weight=5, opacity=0.9).add_to(route)

    # Raw GPS observations: thin trace plus (sampled) dots.
    layer = folium.FeatureGroup(name="GPS observations").add_to(m)
    folium.PolyLine(gps, color=GPS_COLOR, weight=1.5, opacity=0.5).add_to(layer)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=2.5, color=GPS_COLOR, weight=0, fill=True,
                            fill_opacity=0.7, tooltip=f"GPS point {i + 1}").add_to(layer)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  .ps-title {font-size: 2.1rem; font-weight: 750; margin-bottom: 0;}
  .ps-sub {color: #6b7280; margin-top: .1rem;}
  .legend-dot {display:inline-block; width:12px; height:12px; border-radius:50%; margin:0 6px 0 14px;}
  div[data-testid="stMetric"] {background: rgba(127,127,127,.06); border: 1px solid rgba(127,127,127,.18);
                               border-radius: 10px; padding: 10px 14px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="ps-title">🛰️ PathSnap</p>', unsafe_allow_html=True)
st.markdown('<p class="ps-sub">Reconstructing the road route behind noisy GPS: '
            'candidate roads → ML scoring → HMM + Viterbi decoding</p>', unsafe_allow_html=True)

# ---- sidebar: connection to Laptop 1 -----------------------------------------------------------
with st.sidebar:
    st.header("Backend connection")
    api_url = st.text_input("API base URL (Laptop 1)", value=default_api_url(),
                            help="For the two-laptop demo: http://<Laptop 1 IP>:8000")
    try:
        health = api_get(api_url, "/health")
        st.success(f"Connected · {health['n_trips']} trips available")
        mode = health["mode"]
        st.caption(f"**Scorer:** {mode['scorer']}")
        st.caption(f"**Decoder:** {mode['decoder']}")
        st.caption(f"**Decoder params:** {mode['decoder_params']}")
    except Exception as e:
        health = None
        st.error("Cannot reach the API")
        st.caption(f"{type(e).__name__}: {e}")
        st.caption("Check: API running on Laptop 1 with --host 0.0.0.0, same Wi-Fi, correct IP, firewall.")
    if st.button("Reconnect", use_container_width=True):
        st.rerun()

if health is None:
    st.info("Start the API on Laptop 1 and set its address in the sidebar.")
    st.stop()

if health["mode"]["is_mock"]:
    st.warning("**MOCK components active.** The backend is using placeholder components, so these numbers "
               "test the application only. They are NOT real ML results.")

# ---- trip selection --------------------------------------------------------------------------
try:
    trips = api_get(api_url, "/trips")["trips"]
except Exception as e:
    st.error(f"Could not load trips: {e}")
    st.stop()

c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
trip_id = c1.selectbox(f"Evaluation trip ({len(trips)} available)", trips)
run = c2.button("▶  Run map matching", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Laptop 1 is processing trip {trip_id}: candidates → scoring → HMM decoding…"):
        try:
            st.session_state["result"] = api_predict(api_url, trip_id)
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if not res:
    st.caption("Pick a trip and press **Run map matching**.")
    st.stop()

# ---- results ---------------------------------------------------------------------------------
st.subheader(f"Trip {res['trip_id']}")
m = res.get("metrics") or {}
hmm, clf = m.get("hmm", {}), m.get("classifier_only", {})

k1, k2, k3, k4 = st.columns(4)
if m.get("available"):
    k1.metric("Top-1 accuracy (HMM)", pct(hmm["top1_accuracy"]),
              delta=f"{(hmm['top1_accuracy'] - clf['top1_accuracy']) * 100:+.1f} pts vs classifier-only")
    k2.metric("Macro F1 (HMM)", pct(hmm["macro_f1"]),
              delta=f"{(hmm['macro_f1'] - clf['macro_f1']) * 100:+.1f} pts vs classifier-only")
    k3.metric("Route continuity (HMM)", pct(hmm["route_continuity"]),
              delta=f"{(hmm['route_continuity'] - clf['route_continuity']) * 100:+.1f} pts vs classifier-only")
else:
    k1.metric("Top-1 accuracy", "n/a")
    k2.metric("Macro F1", "n/a")
    k3.metric("Route continuity", "n/a")
    st.info(f"Metrics unavailable: {m.get('reason', 'not returned by the API')}")
k4.metric("Server processing time", f"{res['request_s']:.2f} s", delta=f"{res['n_observations']:,} GPS points",
          delta_color="off")

map_col, info_col = st.columns([2.2, 1])
with map_col:
    st.markdown(f'<span class="legend-dot" style="background:{ROUTE_COLOR}"></span>Predicted route'
                f'<span class="legend-dot" style="background:{GPS_COLOR}"></span>GPS observations',
                unsafe_allow_html=True)
    st_folium(build_map(res), height=560, use_container_width=True, returned_objects=[],
              key=f"map_{res['trip_id']}")

with info_col:
    st.markdown("**Classifier-only vs HMM + Viterbi**")
    if m.get("available"):
        st.table({
            "Metric": ["Top-1 accuracy", "Macro F1", "Route continuity"],
            "Classifier-only": [pct(clf["top1_accuracy"]), pct(clf["macro_f1"]), pct(clf["route_continuity"])],
            "HMM + Viterbi": [pct(hmm["top1_accuracy"]), pct(hmm["macro_f1"]), pct(hmm["route_continuity"])],
        })
        st.caption(f"Scored on {m['n_scored_observations']:,} of {m['n_observations']:,} GPS points "
                   f"({m['n_excluded_unreliable_labels']} with unreliable reference labels excluded). "
                   "Ground truth stays on Laptop 1.")
    else:
        st.caption("No metrics for this trip.")

    st.markdown("**Processing time on Laptop 1**")
    t = res["timing"]
    st.table({
        "Stage": ["Road graph", "Candidates + features", "Scoring", "HMM decoding", "Pipeline total", "Whole request"],
        "Seconds": [f"{t['load_graph_s']:.3f}", f"{t['candidates_s']:.3f}", f"{t['scoring_s']:.3f}",
                    f"{t['decoding_s']:.3f}", f"{t['total_s']:.3f}", f"{res['request_s']:.3f}"],
    })
    st.caption(f"{res['n_candidate_rows']:,} candidate road segments · "
               f"{res['n_obs_without_candidates']} GPS points without candidates · "
               f"{len(res['route_segments']):,} distinct predicted arcs")

with st.expander("Model and decoder used for this result"):
    st.json(res["mode"])
