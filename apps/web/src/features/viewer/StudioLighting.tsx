import { Environment, Lightformer, SoftShadows } from '@react-three/drei';
import { useViewerStore } from '../../store/viewerStore';

/** The cubemap is captured once, then three.js PMREM supplies diffuse/specular IBL at every roughness. */
export function StudioLighting() {
  const quality = useViewerStore((state) => state.quality);
  const high = quality === 'high';
  return (
    <>
      <Environment frames={1} resolution={128} background={false} environmentIntensity={0.8}>
        <color attach="background" args={['#343c44']} />
        <Lightformer
          position={[3, 4, 3]}
          target={[0, 1, 0]}
          scale={[4, 5, 1]}
          intensity={4}
          color="#fff2e4"
        />
        <Lightformer
          position={[-4, 2, 2]}
          target={[0, 1, 0]}
          scale={[3, 4, 1]}
          intensity={2.5}
          color="#e7f1ff"
        />
        <Lightformer
          position={[1, 3, -4]}
          target={[0, 1, 0]}
          scale={[2, 4, 1]}
          intensity={4}
          color="#d8eaff"
        />
        <Lightformer position={[0, 5, 0]} target={[0, 0, 0]} scale={[4, 4, 1]} intensity={1.5} />
      </Environment>
      <SoftShadows size={35} samples={high ? 12 : 6} />
      <hemisphereLight args={['#dce8f5', '#51545a', 0.35]} />
      <directionalLight
        key={quality}
        position={[3, 4, 4]}
        color="#fff0e2"
        intensity={3.2}
        castShadow
        shadow-mapSize={[high ? 2048 : 1024, high ? 2048 : 1024]}
        shadow-intensity={0.7}
        shadow-bias={-0.00015}
        shadow-normalBias={0.006}
        shadow-camera-left={-1.8}
        shadow-camera-right={1.8}
        shadow-camera-top={2.1}
        shadow-camera-bottom={-1.1}
        shadow-camera-near={0.5}
        shadow-camera-far={10}
      />
      <directionalLight position={[-3, 2, 3]} intensity={1.1} color="#e2edff" />
      <directionalLight position={[1, 3, -3]} intensity={2} color="#c9e1ff" />
    </>
  );
}
