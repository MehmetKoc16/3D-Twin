import sys, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glbio import read_glb
scan, out, png = sys.argv[1:4]
v = read_glb(scan).prims[0].positions.astype(float)
res = pickle.load(open(out + "/fit.pkl", "rb")); t = res.posed_vertices
fig, axs = plt.subplots(1, 2, figsize=(14, 9))
for ax, (a, b) in zip(axs, [(0, 1), (2, 1)]):
    ax.scatter(v[::3, a], v[::3, b], c="0.6", s=0.6)
    ax.scatter(t[::3, a], t[::3, b], c="r", s=0.6)
    ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(png, dpi=55)
