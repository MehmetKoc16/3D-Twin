# Pose library

Run `node apps/web/scripts/generate-poses.mjs` from the repository root to regenerate
`public/assets/poses/*.json`. This wave does not own package scripts.

`apps/web/scripts/poseSpecs.mjs` contains ordered labels and intent-level directions
and angles. The generator reads the real rig and joint points, solves each bone's
world direction, then converts it to a parent-local quaternion. All rest frames
are world-aligned. The A-pose is the MakeHuman rest pose and has no rotations.
The hip pose solves the elbow from arm lengths, target wrist, and an outward
elbow preference. Output quaternions have six decimals; identities are omitted.

Hands (`orientHand`): the rest hand's palm normal is `fingers x (pinky -> index)` for the left hand and its negation for the right
hand (a mirror image flips the cross product); targets are written for the left hand and mirrored for the right. The hand
axis is the visible wrist-to-middle-knuckle axis, not the hand bone's tail (which points 25-40 deg away from it). The 53-bone
rig has no twist bones, so a hand roll about the forearm axis would be absorbed by the few cm of the wrist where the weights
hand over from `lowerarm` to `hand` (the ring shrinks to `cos(roll / 2)`, the "candy wrapper"). `balanceForearmRoll` therefore rolls
`lowerarm` by half of the difference between wrist roll and elbow roll, keeping the hand's world orientation; `poses.test.ts`
bounds every hand / lowerarm roll (30 deg) and skins the real mannequin to check the wrist ring. `tools/twin-lab/rig/wrist_qa.mjs`
measures the same on any rigged GLB (the twin).

`PoseDriver` fetches each selected pose once, validates the JSON, and blends from
the skeleton's current rotations over 350 ms. It reapplies the target every
frame, including after rest skeleton rebuilds. Mount it inside `<Canvas>`
alongside `<Avatar />`; `<Avatar />` must register the live skeleton in
`useAvatarRuntimeStore` and temporarily set identity rotations while calculating
new inverse bind matrices during each rest rebuild.

## Runtime adaptation (hands-on-hips)

Static wrist targets (`wristX = 0.19 m`) sink into wide female hips (up to -1.71 cm penetration on heavy female)
and float on slender male hips (+4.63 cm clearance). `adaptHandsOnHipsPose` adapts the pose dynamically at runtime:

1. **Mesh surface scan (`computeHipHalfWidth`):** Measures the lateral half-width of the torso/pelvis surface
   in the contact zone around target wrist height (`hip.y + 0.17 m`) and forward depth (`hip.z + 0.065 m`).
   Mesh surface vertices are scanned rather than skeletal joint points because joint heads (`thigh_l`) remain deep
   inside the pelvic bone and shift minimally, whereas soft tissue adds up to ~3.5 cm of lateral width on female/heavy
   shapes. Both left (+X) and right (-X) vertices are tested to ensure symmetric half-width calculation.
2. **Wrist target calibration:** Sets `wristX = 0.19 + (hipHalfWidth - neutralHipHalfWidth)`, clamped to `[0.14, 0.26] m`.
   On the neutral mannequin (`neutralHipHalfWidth = 0.157 m`), delta is 0, preserving the baseline ~1.7 cm clearance and the
   realistic twin's 1.24 cm clearance. Across all body shapes (male/female, thin/average/heavy), hand clearance remains strictly
   within 0.5–2.0 cm with zero triangle intersections.
3. **Outward two-bone IK & roll balancing:** Re-solves the arm chain (shoulder, outward elbow, wrist) and aligns the palm
   normal to `[-1, -0.2, 0]` with `balanceForearmRoll` twist sharing, ensuring roll stays bounded under 30 degrees and the
   linear blend skinning wrist cross-section remains >= 95% open.
4. **Performance:** Pose adaptation runs once on pose selection or when `restVersion` bumps (after body slider adjustments),
   avoiding any heavy vertex iteration or matrix calculations in `useFrame`.

To add a pose, add its id to `poseSpecs.mjs`, implement its intent in
`generate-poses.mjs`, add it to `poseStore` and `PoseBar`, add both translation
keys, regenerate JSON, and add a forward-kinematics assertion.
