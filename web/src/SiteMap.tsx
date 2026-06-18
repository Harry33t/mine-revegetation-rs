import { useEffect, useState } from "react";
import { loadSites, type Site } from "./data";

// Simple offline Australia map: an embedded (simplified) coastline + study-site
// markers projected from lon/lat. No tile service / API key needed.
const LON0 = 112, LON1 = 154, LAT0 = -44, LAT1 = -9;
const W = 440, H = 380;

function proj(lon: number, lat: number): [number, number] {
  return [((lon - LON0) / (LON1 - LON0)) * W, ((LAT1 - lat) / (LAT1 - LAT0)) * H];
}

// simplified mainland outline (lon,lat), clockwise
const AUS: [number, number][] = [
  [126.9, -13.7], [130.8, -12.4], [137, -12], [135.5, -15], [137, -16],
  [140.8, -17.5], [141.5, -12.5], [142.5, -10.9], [145.3, -15], [146, -18.5],
  [149, -21], [151, -24], [153.0, -25.9], [153.6, -28.6], [152.5, -32],
  [151.2, -33.9], [150, -37.5], [147, -38.8], [144.5, -38.4], [141, -38.3],
  [139, -35.5], [138.6, -34.9], [137.5, -35], [136, -35], [135.7, -34.7],
  [134, -33], [129, -31.7], [125, -32.5], [120, -33.8], [115.1, -34.4],
  [115.7, -31.5], [114.9, -29], [113.4, -26], [114, -22], [116, -20.5],
  [118.6, -20.3], [121, -19.5], [122.2, -18], [124, -16],
];
const TAS: [number, number][] = [[145, -40.7], [148.3, -40.7], [148, -43.6], [146, -43.5]];

function pathFrom(pts: [number, number][]): string {
  return pts.map((p, i) => `${i ? "L" : "M"}${proj(p[0], p[1]).map((v) => v.toFixed(1)).join(",")}`).join(" ") + " Z";
}

function shortName(name: string): string {
  return name.split("_").map((s) => s[0].toUpperCase() + s.slice(1)).join(" ");
}

export default function SiteMap() {
  const [sites, setSites] = useState<Site[]>([]);
  const [hover, setHover] = useState<string | null>(null);
  useEffect(() => {
    loadSites().then((s) => setSites(s.filter((x) => x.track === "satellite"))).catch(() => {});
  }, []);
  return (
    <div className="sitemap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Study-site map of Australia">
        <path className="aus" d={pathFrom(AUS)} />
        <path className="aus" d={pathFrom(TAS)} />
        {sites.map((s) => {
          const [x, y] = proj(s.lon, s.lat);
          const on = hover === s.name;
          const leftSide = x > W * 0.62; // east-coast labels go to the left to avoid clipping
          const label = `${shortName(s.name)}${s.mineral ? ` · ${s.mineral.replace("_", " ")}` : ""}`;
          return (
            <g key={s.name} className="site" onMouseEnter={() => setHover(s.name)} onMouseLeave={() => setHover(null)}>
              <circle cx={x} cy={y} r={on ? 8 : 5.5} className="pin" />
              <circle cx={x} cy={y} r={on ? 8 : 5.5} className="pin-ring" />
              <text x={leftSide ? x - 11 : x + 11} y={y + 4} textAnchor={leftSide ? "end" : "start"}
                className={"lbl" + (on ? " on" : "")}>
                {label}
              </text>
            </g>
          );
        })}
        <text x={12} y={H - 12} className="mapnote">Australian rehabilitating mine sites · Sentinel-2</text>
      </svg>
    </div>
  );
}
