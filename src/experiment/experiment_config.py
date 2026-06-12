import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
lab_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
config_path = os.path.join(lab_root, "config", "experiment_conditions.json")

_DEFAULTS = {
    "L_fg": 15.0,
    "L_bg": 15.0,
    "L_ref": 30.0,
    "VISUAL_ANGLE_DEG": 7.9,
    "DISTANCE_FG": 50,
    "DISTANCE_BG": 150,
    "BG_COLOR": "black"
}


def get_config():
    cfg = dict(_DEFAULTS)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    cfg.update(data)
    except Exception:
        pass
    return cfg


def get(key, default=None):
    return get_config().get(key, default)
