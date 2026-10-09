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

To add a pose, add its id to `poseSpecs.mjs`, implement its intent in
`generate-poses.mjs`, add it to `poseStore` and `PoseBar`, add both translation
keys, regenerate JSON, and add a forward-kinematics assertion.
