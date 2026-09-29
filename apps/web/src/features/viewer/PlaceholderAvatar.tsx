import { Quaternion, Vector3 } from 'three';
import type { BodyParams } from '@dt/avatar-core';
import { useViewerStore } from '../../store/viewerStore';
import { footLengthCmFromShoe } from '../body-panel/shoeSize';

function Limb({ from, to, radius, color }: { from: [number, number, number]; to: [number, number, number]; radius: number; color: string }) {
  const start = new Vector3(...from);
  const end = new Vector3(...to);
  const delta = end.clone().sub(start);
  const midpoint = start.clone().add(end).multiplyScalar(0.5);
  const quaternion = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), delta.normalize());
  return <mesh position={midpoint} quaternion={quaternion} castShadow>
    <capsuleGeometry args={[radius, Math.max(0.01, delta.length() - radius * 2), 8, 16]} />
    <meshStandardMaterial color={color} roughness={0.7} />
  </mesh>;
}

function Segment({ from, to, radius, color }: { from: [number, number, number]; to: [number, number, number]; radius: number; color: string }) {
  return <Limb from={from} to={to} radius={radius} color={color} />;
}

export function PlaceholderAvatar({ params, poseId }: { params: BodyParams; poseId: string }) {
  const requestPoint = useViewerStore((state) => state.requestPoint);
  const h = params.heightCm / 100;
  const shoulder = (params.shoulderCm ?? 45) / 200 * (0.94 + 0.06 * params.gender);
  const chest = (params.chestCm ?? 100) / 100;
  const waist = (params.waistCm ?? 85) / 100;
  const hip = (params.hipCm ?? 100) / 100;
  const mass = Math.max(0.8, Math.min(1.2, params.weightKg / 75));
  const upperRadius = Math.max(0.12, chest / (2 * Math.PI)) * mass * (0.92 + 0.08 * params.gender);
  const waistRadius = Math.max(0.11, waist / (2 * Math.PI)) * mass;
  const hipRadius = Math.max(0.12, hip / (2 * Math.PI)) * mass * (1.08 - 0.08 * params.gender);
  const footLength = footLengthCmFromShoe(params.shoe.system, params.shoe.size) / 100;
  const armAngle = poseId === 't-pose' ? 0 : poseId === 'a-pose' ? 0.55 : poseId === 'relaxed' ? 1.25 : poseId === 'hands-on-hips' ? 0.95 : 1.4;
  const armReach = h * 0.34;
  const shoulderY = h * 0.79;
  const legY = h * 0.49;
  const skin = '#c9a88d';
  const clothing = '#89a7ac';
  const leg = '#454f5b';
  const armEnd = (side: number): [number, number, number] => poseId === 'hands-on-hips'
    ? [side * hipRadius, h * 0.52, 0.04]
    : [side * (shoulder + armReach * Math.cos(armAngle)), shoulderY - armReach * Math.sin(armAngle), poseId === 'walk' ? side * 0.12 : 0];
  return <group rotation={[0, poseId === 'side' ? Math.PI / 2 : 0, 0]}
    onDoubleClick={(event) => { event.stopPropagation(); requestPoint([event.point.x, event.point.y, event.point.z]); }}>
    <mesh position={[0, h * 0.65, 0]} scale={[upperRadius * 1.65, h * 0.17, upperRadius]} castShadow>
      <sphereGeometry args={[1, 28, 20]} /><meshStandardMaterial color={clothing} roughness={0.85} />
    </mesh>
    <mesh position={[0, h * 0.52, 0]} scale={[waistRadius * 1.2, h * 0.09, waistRadius]} castShadow>
      <sphereGeometry args={[1, 24, 16]} /><meshStandardMaterial color={clothing} />
    </mesh>
    <mesh position={[0, h * 0.46, 0]} scale={[hipRadius * 1.4, h * 0.075, hipRadius]} castShadow>
      <sphereGeometry args={[1, 24, 16]} /><meshStandardMaterial color={leg} />
    </mesh>
    <mesh position={[0, h * 0.86, 0]} scale={[0.055, h * 0.055, 0.055]} castShadow>
      <cylinderGeometry args={[1, 1, 2, 16]} /><meshStandardMaterial color={skin} />
    </mesh>
    <mesh position={[0, h * 0.94, 0]} scale={[0.095 + (1 - params.gender) * 0.005, h * 0.065, 0.085]} castShadow>
      <sphereGeometry args={[1, 28, 20]} /><meshStandardMaterial color={skin} roughness={0.8} />
    </mesh>
    {([-1, 1] as const).map((side) => <group key={side}>
      <Segment from={[side * shoulder, shoulderY, 0]} to={armEnd(side)} radius={0.055 * mass} color={skin} />
      <Segment from={[side * hipRadius * 0.7, legY, 0]} to={[side * hipRadius * 0.58, h * 0.07, poseId === 'walk' ? side * 0.14 : 0]} radius={0.075 * mass} color={leg} />
      <mesh position={[side * hipRadius * 0.58, h * 0.025, footLength * 0.2 + (poseId === 'walk' ? side * 0.14 : 0)]} scale={[0.09, 0.045, footLength * 0.65]} castShadow>
        <sphereGeometry args={[1, 16, 12]} /><meshStandardMaterial color="#d9e1df" />
      </mesh>
    </group>)}
  </group>;
}

