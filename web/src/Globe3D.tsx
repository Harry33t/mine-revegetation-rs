import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { loadSites, type Site } from "./data";

// lat/lon (deg) -> point on a sphere of radius r.
function llToVec(lat: number, lon: number, r: number): [number, number, number] {
  const phi = ((90 - lat) * Math.PI) / 180;
  const theta = ((lon + 180) * Math.PI) / 180;
  return [-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta)];
}

function Marker({ site }: { site: Site }) {
  const pos = llToVec(site.lat, site.lon, 1.015);
  const color = site.track === "bridge" ? "#46c4c4" : "#ffd24a";
  const [hover, setHover] = useState(false);
  return (
    <group position={pos}>
      <mesh onPointerOver={() => setHover(true)} onPointerOut={() => setHover(false)}>
        <sphereGeometry args={[hover ? 0.04 : 0.025, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.4} />
      </mesh>
      {hover && (
        <Html distanceFactor={6} style={{ pointerEvents: "none" }}>
          <div className="globe-label">{site.label}</div>
        </Html>
      )}
    </group>
  );
}

function Globe({ sites }: { sites: Site[] }) {
  return (
    <group rotation={[0.2, 0, 0]}>
      <mesh>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial color="#0f2a1c" roughness={0.9} metalness={0.1} />
      </mesh>
      <mesh scale={1.004}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial color="#2f6b4a" wireframe transparent opacity={0.35} />
      </mesh>
      {sites.map((s) => (
        <Marker key={s.name} site={s} />
      ))}
    </group>
  );
}

export default function Globe3D() {
  const [sites, setSites] = useState<Site[]>([]);
  useEffect(() => {
    loadSites().then(setSites).catch(() => {});
  }, []);
  return (
    <div className="r3f globe">
      <Canvas camera={{ position: [0, 0.6, 3], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[5, 3, 5]} intensity={1.1} />
        <Globe sites={sites} />
        <OrbitControls autoRotate autoRotateSpeed={0.9} enableZoom={false} enablePan={false} />
      </Canvas>
    </div>
  );
}
