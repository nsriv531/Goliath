# ai/choice_logging.py
import json
import os
from datetime import datetime
from typing import Dict, Optional

LOG_FILE = os.path.join(os.path.dirname(__file__), "run_logs.jsonl")

def start_new_run() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

def log_choice(run_id: str, scene_id: str, choice_key: str, meta: Optional[Dict] = None) -> None:
    event = {
        "run_id": run_id,
        "scene_id": scene_id,
        "choice_key": choice_key,
        "meta": meta or {},
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
