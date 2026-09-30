import sys, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glbio import read_glb
from mh import MHModel
from rigfit import transfer_weights
scan, out, png, bone = sys.argv[1:5]
sc = read_glb(scan); v = sc.prims[0].positions.astype(float); f = sc.prims[0].indices.astype(int)
key = np.round(v*1e5).astype(np.int64); _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
uv_ = v[first]; uf = inv[f]
m = MHModel(); res = pickle.load(open(out + "/fit.pkl", "rb"))
W, _ = transfer_weights(m, res.posed_vertices, uv_, uf)
ids = [i for n, i in m.bone_index.items() if n.endswith("_l") and n.split("_")[0] in bone.split(",")]
hw = W[:, ids].sum(1)
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
for ax, (a, b) in zip(axs, [(0, 1), (2, 1), (0, 2)]):
    ax.scatter(uv_[::2, a], uv_[::2, b], c="0.8", s=0.5)
    s = hw > 0.3; ax.scatter(uv_[s, a], uv_[s, b], c=hw[s], cmap="autumn", s=2)
    ax.set_aspect("equal")
    if a == 0 and b == 1: ax.set_xlim(0, 0.4); ax.set_ylim(0.5, 1.4)
    if a == 2: ax.set_ylim(0.5, 1.4); ax.set_xlim(-0.2, 0.2)
    if b == 2: ax.set_xlim(0, 0.4); ax.set_ylim(-0.2, 0.2)
plt.tight_layout(); plt.savefig(png, dpi=55)
