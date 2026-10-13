// Intent uses world axes: +Y up, +Z forward, character left +X.
export const poseSpecs = [
  {
    id: 't-pose',
    label: { tr: 'T-Poz', en: 'T-Pose' },
    intent: { upperArm: 'horizontal outward', forearm: 'horizontal outward', palm: 'down' },
  },
  {
    id: 'a-pose',
    label: { tr: 'A-Poz', en: 'A-Pose' },
    intent: { body: 'unaltered MakeHuman A-pose' },
  },
  {
    id: 'relaxed',
    label: { tr: 'Rahat', en: 'Relaxed' },
    intent: {
      clavicleDropDeg: 12,
      upperArmOutwardDeg: 8,
      elbowFlexDeg: 8,
      upperArmForward: { l: 0.02, r: 0.02 },
    },
  },
  {
    id: 'hands-on-hips',
    label: { tr: 'Eller belde', en: 'Hands on hips' },
    intent: {
      clavicleDropDeg: 12,
      // Wrist target relative to the thigh joint: the hand rests on the iliac crest / waist, not on the hip joint level
      // (the hip-level target sank the palm into the thigh and pelvis). The palm normal is world -X (toward the body)
      // tilted down by 0.2 so the palm follows the slope of the hip. Calibrated on the neutral mannequin (about 1.7 cm
      // clearance), the private twin (about 1.2 cm) and slimmer / heavier MakeHuman shapes; a static pose cannot follow
      // very wide female hips.
      wristX: 0.19,
      wristAboveHipM: 0.17,
      wristForwardM: 0.065,
      palmNormal: [-1, -0.2, 0],
      elbow: 'outward two-bone IK',
    },
  },
  {
    id: 'walk',
    label: { tr: 'Yürüyüş', en: 'Walking step' },
    intent: {
      clavicleDropDeg: 12,
      upperArmOutwardDeg: 12,
      elbowFlexDeg: 8,
      upperArmForward: { l: -0.22, r: 0.3 },
      leftThighForwardDeg: 25,
      rightThighBackDeg: 15,
      rightKneeFlexDeg: 20,
      leftKneeFlexDeg: 20,
    },
  },
  {
    id: 'side',
    label: { tr: 'Yan profil', en: 'Side profile' },
    intent: {
      body: 'relaxed',
      rootYawDeg: 90,
      clavicleDropDeg: 12,
      upperArmOutwardDeg: 8,
      elbowFlexDeg: 8,
      upperArmForward: { l: 0.02, r: 0.02 },
    },
  },
];
