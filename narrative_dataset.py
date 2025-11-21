# ai/narrative_dataset.py
import json, os
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset

LOG_FILE = os.path.join(os.path.dirname(__file__), "run_logs.jsonl")

CHOICE_CATEGORY = {
    "attack_guard": "AGGRESSIVE",
    "charge_beast": "AGGRESSIVE",
    "threaten_merchant": "AGGRESSIVE",
    "talk_guard": "DIPLOMATIC",
    "negotiate": "DIPLOMATIC",
    "comfort_villager": "DIPLOMATIC",
    "sneak_past": "CAUTIOUS",
    "hide": "CAUTIOUS",
    "retreat": "CAUTIOUS",
}

CATEGORIES = ["AGGRESSIVE", "DIPLOMATIC", "CAUTIOUS"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

def load_runs():
    runs = defaultdict(list)
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(LOG_FILE)
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                runs[ev["run_id"]].append(ev)
    return runs

def extract_features_and_labels(runs):
    X, y = [], []
    for run_id, events in runs.items():
        counts = Counter()
        for ev in events:
            ck = ev["choice_key"]
            cat = CHOICE_CATEGORY.get(ck)
            if cat:
                counts[cat] += 1

        if not counts:
            continue

        agg = counts["AGGRESSIVE"]
        dip = counts["DIPLOMATIC"]
        cau = counts["CAUTIOUS"]

        label = max(counts, key=lambda k: counts[k])
        X.append([agg, dip, cau])
        y.append(CATEGORY_TO_IDX[label])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class NarrativeStyleDataset(Dataset):
    def __init__(self):
        runs = load_runs()
        self.X, self.y = extract_features_and_labels(runs)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
