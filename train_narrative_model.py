# ai/train_narrative_model.py
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import os

from .narrative_dataset import NarrativeStyleDataset
from .narrative_model import NarrativeStyleModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "narrative_style_model.pt")

def get_loaders(batch=8):
    ds = NarrativeStyleDataset()
    val = max(1, int(len(ds)*0.2))
    train = len(ds)-val
    train_ds, val_ds = random_split(ds, [train, val])
    return DataLoader(train_ds, batch_size=batch, shuffle=True), DataLoader(val_ds, batch_size=batch)

def train(epochs=20):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_l, val_l = get_loaders()
    model = NarrativeStyleModel().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        for X,y in train_l:
            X,y = X.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()
