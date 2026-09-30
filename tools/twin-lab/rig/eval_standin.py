"""Compare the rigged stand-in against ground truth (rest joints, pose)."""
import json, sys, numpy as np, os
d = sys.argv[1] if len(sys.argv) > 1 else ".cache/standin"
o = sys.argv[2] if len(sys.argv) > 2 else ".cache/standin_out"
t = json.load(open(os.path.join(d, "truth.json")))
from mh import MHModel
m = MHModel()
est = np.load(os.path.join(o, "rest_heads.npy"))
truth = np.array([t["rest_heads"][n] for n in m.bone_names])
err = np.linalg.norm(est - truth, axis=1) * 100
print("rest joint error cm: mean %.2f median %.2f max %.2f" % (err.mean(), np.median(err), err.max()))
for k in ["pelvis","spine_03","neck_01","head","upperarm_l","lowerarm_l","hand_l","thigh_l","calf_l","foot_l","upperarm_r","hand_r","calf_r"]:
    i = m.bone_index[k]; print("  %-12s %.2f cm" % (k, err[i]))
r = json.load(open(os.path.join(o, "rig_report.json")))
print(json.dumps(r["fit"]["pose_deg"]))
print("truth pose", t["pose_deg"])
