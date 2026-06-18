import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { loadHeightmap, type HeightMap } from "./data";

function Surface({ hm }: { hm: HeightMap }) {
  const geom = useMemo(() => {
    const { rows, cols, data, zmax } = hm;
    const aspect = rows / cols;
    const g = new THREE.PlaneGeometry(4, 4 * aspect, cols - 1, rows - 1);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const colors: number[] = [];
    const zscale = 1.4 / (zmax || 1);
    for (let i = 0; i < pos.count; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const h = data[r]?.[c] ?? 0;
      pos.setZ(i, h * zscale);
      const t = Math.min(1, h / (zmax || 1));
      const col = new THREE.Color().setHSL(0.33, 0.55, 0.18 + 0.5 * t); // bare->canopy green ramp
      colors.push(col.r, col.g, col.b);
    }
    g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    g.computeVertexNormals();
    return g;
  }, [hm]);

  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2.4, 0, 0]}>
      <meshStandardMaterial vertexColors flatShading side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function Canopy3D() {
  const [hm, setHm] = useState<HeightMap | null>(null);
  useEffect(() => {
    loadHeightmap().then(setHm).catch(() => {});
  }, []);
  if (!hm) return <div className="r3f loading">loading 3D surface…</div>;
  return (
    <div className="r3f">
      <Canvas camera={{ position: [0, 2.4, 3.4], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 6, 2]} intensity={1.2} />
        <Surface hm={hm} />
        <OrbitControls autoRotate autoRotateSpeed={1.1} enableZoom enablePan={false} minDistance={2.5} maxDistance={6} />
      </Canvas>
    </div>
  );
}
