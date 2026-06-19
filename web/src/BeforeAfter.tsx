import { useRef, useState } from "react";
import { asset } from "./data";

// Swipe slider comparing the same ground in 2019 vs 2024 over the recovery window.
export default function BeforeAfter() {
  const [pos, setPos] = useState(50);
  const [mode, setMode] = useState<"rgb" | "ndvi">("rgb");
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFrom = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };

  const before = asset(mode === "rgb" ? "ba_rgb_2019.png" : "ba_ndvi_2019.png");
  const after = asset(mode === "rgb" ? "ba_rgb_2024.png" : "ba_ndvi_2024.png");

  return (
    <div className="ba-wrap">
      <div
        className="ba"
        ref={ref}
        onPointerDown={(e) => { dragging.current = true; setFrom(e.clientX); }}
        onPointerMove={(e) => { if (dragging.current) setFrom(e.clientX); }}
        onPointerUp={() => { dragging.current = false; }}
        onPointerLeave={() => { dragging.current = false; }}
      >
        <img className="ba-img" src={before} alt="2019" draggable={false} />
        <img className="ba-img ba-after" src={after} alt="2024" draggable={false}
          style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <div className="ba-divider" style={{ left: `${pos}%` }}><span className="ba-handle" /></div>
        <span className="ba-tag ba-left">2019</span>
        <span className="ba-tag ba-right">2024</span>
      </div>
      <div className="ba-toggle">
        <button className={mode === "rgb" ? "on" : ""} onClick={() => setMode("rgb")}>true colour</button>
        <button className={mode === "ndvi" ? "on" : ""} onClick={() => setMode("ndvi")}>NDVI</button>
        <span className="ba-hint">drag to compare</span>
      </div>
    </div>
  );
}
