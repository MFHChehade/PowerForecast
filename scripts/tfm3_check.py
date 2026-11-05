import sys, torch, functools
import os; sys.path.insert(0, os.path.expandvars("$WORK/pf_tacc/timesfm3_pkgs"))
from timesfm3.torch import model as M
m = M.TimesFM3Torch.from_pretrained("google/timesfm-3.0-pytorch")
print("hub cfg keys:", list(getattr(m, "_hub_mixin_config", {}).keys()))
print("decode wrapped:", hasattr(M.TimesFM3Torch.decode, "__wrapped__"))
s = M.TimesFM3Torch(**m._hub_mixin_config)
ks, kp = s.state_dict(), m.state_dict()
print("keys equal:", ks.keys() == kp.keys(), len(kp))
bad = [k for k in kp if ks[k].shape != kp[k].shape]
print("shape mismatches:", bad[:5])
print("nparams M:", sum(p.numel() for p in m.parameters())/1e6)
same = [k for k in kp if ks[k].shape == kp[k].shape and torch.equal(ks[k].float(), kp[k].float())]
print("identical tensors scratch vs pretrained:", len(same), same[:8])
x = torch.randn(4, 1, 256)
y = M.TimesFM3Torch.decode.__wrapped__(s, x, horizon=24)
print("grad ok:", y.requires_grad, tuple(y.shape))
