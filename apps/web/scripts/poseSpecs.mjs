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
      wristX: 0.225,
      wristAboveHipM: 0.055,
      wristForwardM: 0.065,
      elbow: 'outward two-bone IK',
    },
  },
  {
    id: 'walk',
    label: { tr: 'Yürüyüş', en: 'Walking step' },
    intent: {
      clavicleDropDeg: 12,
      upperArmOutwardDeg: 8,
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
