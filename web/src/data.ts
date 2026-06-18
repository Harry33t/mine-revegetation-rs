// Loaders + types for artifacts bundled by scripts/export_web_data.py into public/data.

const BASE = import.meta.env.BASE_URL;

export interface TrajectoryPoint {
  year: number;
  mean_ndvi: number;
  veg_cover_frac: number;
  recovery_fraction?: number;
}

export interface LabelEffPoint {
  fraction: number;
  n_train: number;
  miou_mean?: number;
  miou_std?: number;
  miou?: number; // single-run fallback
  per_class_iou?: Record<string, number>;
}

export interface Manifest {
  assets: string[];
  trajectories: string[];
  has_label_efficiency: boolean;
}

async function getJSON<T>(rel: string): Promise<T> {
  const res = await fetch(`${BASE}data/${rel}`);
  if (!res.ok) throw new Error(`failed to load ${rel}`);
  return res.json();
}

export interface Site {
  name: string;
  label: string;
  track: string;
  lon: number;
  lat: number;
  mineral: string;
}

export interface HeightMap {
  rows: number;
  cols: number;
  zmin: number;
  zmax: number;
  data: number[][];
  label: string;
}

export interface TransferMatrix {
  sites: string[];
  matrix: number[][];
  std?: number[][];
  seeds?: number;
}

export const loadManifest = () => getJSON<Manifest>("manifest.json");
export const loadSites = () => getJSON<Site[]>("sites.json");
export const loadHeightmap = () => getJSON<HeightMap>("heightmap.json");
export const loadTransferMatrix = () => getJSON<TransferMatrix>("transfer_matrix.json");

export interface SiteStat { kappa: number; overall_agreement: number; n_pixels: number; }
export interface Validation {
  per_site: Record<string, SiteStat>;
  pooled: SiteStat;
  reference: string;
  scheme: string;
  year: string;
}
export const loadValidation = () => getJSON<Validation>("validation.json");
export const loadTrajectories = () =>
  getJSON<Record<string, TrajectoryPoint[]>>("trajectories.json");
export const loadLabelEfficiency = () => getJSON<LabelEffPoint[]>("label_efficiency.json");

export const asset = (name: string) => `${BASE}assets/${name}`;

// Pretty site label from a trajectory key like "alcoa_huntly_rehab_trajectory".
export function prettySite(key: string): string {
  const name = key.replace(/_rehab_trajectory$|_trajectory$/, "");
  return name
    .split("_")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
}
