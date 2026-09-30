"""Debug: scatter of scan vertices coloured by dominant bone group (posed frame) + fitted template outline."""
import sys, pickle, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glbio import read_glb
from mh import MHModel
from rigfit import transfer_weights, top4
scan, out, png = sys.argv[1:4]
sc = read_glb(scan); v = sc.prims[0].positions.astype(float); f = sc.prims[0].indices.astype(int)
key = np.round(v*1e5).astype(np.int64); _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
uv_ = v[first]; uf = inv[f]
m = MHModel(); res = pickle.load(open(out + "/fit.pkl", "rb"))
W, _ = transfer_weights(m, res.posed_vertices, uv_, uf)
dom = W.argmax(1)
groups = {}
def grp(n):
    b = n.split("_")[0]
    side = n[-1] if n.endswith(("_l", "_r")) else ""
    if b in ("index","middle","ring","pinky","thumb"): b = "fingers"
    return b + side
gnames = sorted({grp(n) for n in m.bone_names}); cmap = plt.get_cmap("tab20")
col = np.array([cmap(gnames.index(grp(m.bone_names[d])) % 20) for d in dom])
fig, axs = plt.subplots(1, 3, figsize=(18, 9))
for ax, (a, b_) in zip(axs, [(0, 1), (2, 1), (0, 2)]):
    ax.scatter(uv_[:, a], uv_[:, b_], c=col, s=1)
    tv = res.posed_vertices[::6]; ax.scatter(tv[:, a], tv[:, b_], c="k", s=0.3, alpha=0.3)
    ax.set_aspect("equal")
axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(i % 20), label=g) for i, g in enumerate(gnames)], fontsize=6, ncol=2)
axs[2].invert_yaxis() if False else None
plt.tight_layout(); plt.savefig(png, dpi=60)
