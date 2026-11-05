"""Deep-learning forecasters trained from scratch per cell.

The network definitions are the historical ``alcf_model_zoo.py`` architectures,
lifted to module level and parameterized by ``(context, horizon)``. The training
loop in :meth:`DeepForecaster.predict` (early stopping, normalization, the 8 GB
GPU / long-context LSTM fallbacks) is verbatim (SEED=42).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster

SEED = 42


def _build_networks(context, horizon):
    import torch
    import torch.nn as nn

    class NLinear(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(context, horizon)
        def forward(self, x):
            last = x[:, -1:]; return self.fc(x - last) + last

    class DLinear(nn.Module):
        def __init__(self, k=25):
            super().__init__(); self.k = k
            self.t = nn.Linear(context, horizon); self.s = nn.Linear(context, horizon)
        def forward(self, x):
            trend = torch.nn.functional.avg_pool1d(x.unsqueeze(1), self.k, 1, self.k // 2).squeeze(1)
            trend = trend[:, :context]
            return self.t(trend) + self.s(x - trend)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(context, 256), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, horizon))
        def forward(self, x):
            return self.net(x)

    class LSTM(nn.Module):
        def __init__(self, hid=64):
            super().__init__(); self.lstm = nn.LSTM(1, hid, 2, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hid, horizon)
        def forward(self, x):
            o, _ = self.lstm(x.unsqueeze(-1)); return self.fc(o[:, -1])

    class PatchTST(nn.Module):
        def __init__(self, patch=24, d=64, heads=4, layers=2):
            super().__init__()
            self.patch = patch; self.stride = patch
            self.npatch = (context - patch) // self.stride + 1
            self.embed = nn.Linear(patch, d)
            enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d * self.npatch, horizon)
        def forward(self, x):
            ps = [x[:, i * self.stride:i * self.stride + self.patch] for i in range(self.npatch)]
            p = torch.stack(ps, 1)
            z = self.tr(self.embed(p))
            return self.head(z.reshape(z.size(0), -1))

    class NBEATS(nn.Module):
        def __init__(self, units=128, blocks=3):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(blocks):
                self.blocks.append(nn.Sequential(
                    nn.Linear(context, units), nn.ReLU(), nn.Linear(units, units), nn.ReLU(),
                    nn.Linear(units, context + horizon)))
        def forward(self, x):
            res = x; fc = 0.0
            for b in self.blocks:
                o = b(res); back, fore = o[:, :context], o[:, context:]
                res = res - back; fc = fc + fore
            return fc

    return {"NLinear": NLinear, "DLinear": DLinear, "MLP": MLP,
            "LSTM": LSTM, "PatchTST": PatchTST, "NBEATS": NBEATS}


class DeepForecaster(Forecaster):
    """NLinear, DLinear, MLP, PatchTST, LSTM, NBEATS — trained per cell with early stopping."""

    family = "dl"

    def predict(self, split, horizon):
        import torch
        import torch.nn as nn
        name = self.name
        torch.manual_seed(SEED); np.random.seed(SEED)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # cuDNN LSTM reserves a workspace that scales with sequence length, not batch,
        # and OOMs an 8 GB GPU past ~2k steps; run it on CPU (32 GB RAM) at long context.
        context_len = split["Xtr"].shape[1]
        if name == "LSTM" and context_len > 2048:
            dev = "cpu"
        Xtr, Ytr, Xva, Yva, Xte = split["Xtr"], split["Ytr"], split["Xva"], split["Yva"], split["Xte"]
        context = Xtr.shape[1]
        # normalize by train stats
        mu, sd = Xtr.mean(), Xtr.std() + 1e-6

        def nz(a):
            return ((a - mu) / sd).astype(np.float32)

        builders = _build_networks(context, horizon)
        net = builders[name]().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.L1Loss()

        max_epochs = 80
        # CPU fallback (long-context LSTM) is slow per step; subsample train windows and
        # cap epochs to keep each cell tractable without changing the protocol materially.
        if dev == "cpu" and len(Xtr) > 2000:
            rng = np.random.RandomState(SEED)
            sel = rng.choice(len(Xtr), 2000, replace=False)
            Xtr, Ytr = Xtr[sel], Ytr[sel]
            max_epochs = 30
        xt = torch.tensor(nz(Xtr)); yt = torch.tensor((Ytr - mu) / sd)
        xv = torch.tensor(nz(Xva)); yv = torch.tensor((Yva - mu) / sd)
        xe = torch.tensor(nz(Xte))
        bs = 256
        # LSTM holds full-sequence activations; shrink batch hard at long context (8 GB GPU).
        if name == "LSTM" and context > 1024:
            bs = 16
        best, best_state, patience, bad = 1e9, None, 8, 0
        for epoch in range(max_epochs):
            net.train(); perm = torch.randperm(len(xt))
            for i in range(0, len(xt), bs):
                idx = perm[i:i + bs]
                xb, yb = xt[idx].to(dev), yt[idx].to(dev)
                opt.zero_grad(); l = loss_fn(net(xb), yb); l.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                if len(xv):
                    vl = loss_fn(net(xv.to(dev)), yv.to(dev)).item()
                else:
                    vl = l.item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        preds = []
        ebs = bs if name == "LSTM" else min(512, bs * 4)
        with torch.no_grad():
            for i in range(0, len(xe), ebs):
                preds.append(net(xe[i:i + ebs].to(dev)).cpu().numpy())
        pred = np.vstack(preds) * sd + mu
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()
        return pred.reshape(len(Xte), horizon)
