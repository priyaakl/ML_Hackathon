import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]   # repo root


def load_config():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    env = os.environ.get("PATHSNAP_DATA")
    if env:
        cfg["paths"]["data_root"] = env
    cfg["paths"]["data_root"] = Path(cfg["paths"]["data_root"]).expanduser()
    cfg["paths"]["processed"] = ROOT / cfg["paths"]["processed"]
    cfg["paths"]["results"] = ROOT / cfg["paths"]["results"]
    return cfg
