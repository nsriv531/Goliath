# ai/ai_integration.py
import os
import torch
from .narrative_model import NarrativeStyleModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "narrative_style_model.pt")

def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = NarrativeStyleModel().to(dev)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    m.eval()
    return m, dev

def predict(agg, dip, cau):
    m,dev = load_model()
    x = torch.tensor([[agg,dip,cau]], dtype=torch.float32).to(dev)
    with torch.no_grad():
        return torch.argmax(m(x), dim=1).item()
