import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import {
  loadLabelEfficiency, loadSelfTraining, loadTrajectories, loadTransferMatrix, loadValidation,
  prettySite, type LabelEffPoint, type SelfTraining, type TrajectoryPoint,
  type TransferMatrix, type Validation,
} from "./data";
import SiteMap from "./SiteMap";
import Canopy3D from "./Canopy3D";
import UncertaintyGallery from "./UncertaintyGallery";
import BeforeAfter from "./BeforeAfter";

const AXIS = "#5d5d57";
const GRID = "#e2e0d6";
const GREEN = "#2e6b4a";
const OCHRE = "#b5651d";
const TIP = { background: "#fff", border: "1px solid #e2e0d6", fontSize: 13 };

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function prettyName(n: string): string {
  return n.split("_").map((s) => s[0].toUpperCase() + s.slice(1)).join(" ");
}

function LabelEffTable({ pts }: { pts: LabelEffPoint[] }) {
  return (
    <table className="data">
      <caption>Validation mIoU vs. fraction of labelled tiles (mean ± s.d. over 3 seeds).</caption>
      <thead>
        <tr><th>Labelled fraction</th><th>Training tiles</th><th>Validation mIoU</th></tr>
      </thead>
      <tbody>
        {pts.map((p) => (
          <tr key={p.fraction}>
            <td>{(p.fraction * 100).toFixed(0)}%</td>
            <td>{p.n_train}</td>
            <td>{(p.miou_mean ?? p.miou ?? 0).toFixed(3)}
              {p.miou_std != null ? ` ± ${p.miou_std.toFixed(3)}` : ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LocoTable({ tm }: { tm: TransferMatrix }) {
  return (
    <table className="data">
      <caption>Leave-one-site-out transfer: mIoU of a model trained on each site (rows) and
        evaluated on each site (columns). Diagonal = within-site.</caption>
      <thead>
        <tr><th>train ↓ / test →</th>{tm.sites.map((s) => <th key={s}>{prettyName(s)}</th>)}</tr>
      </thead>
      <tbody>
        {tm.sites.map((src, i) => (
          <tr key={src}>
            <td>{prettyName(src)}</td>
            {tm.sites.map((tgt, j) => (
              <td key={tgt} className={i === j ? "diag" : ""}>{tm.matrix[i][j].toFixed(2)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function App() {
  const [traj, setTraj] = useState<Record<string, TrajectoryPoint[]>>({});
  const [labelEff, setLabelEff] = useState<LabelEffPoint[] | null>(null);
  const [loco, setLoco] = useState<TransferMatrix | null>(null);
  const [val, setVal] = useState<Validation | null>(null);
  const [self, setSelf] = useState<SelfTraining | null>(null);
  const [site, setSite] = useState<string>("");

  useEffect(() => {
    loadValidation().then(setVal).catch(() => setVal(null));
    loadSelfTraining().then(setSelf).catch(() => setSelf(null));
    loadTrajectories().then((t) => {
      setTraj(t);
      const rehab = Object.keys(t).filter((k) => k.includes("rehab"));
      setSite((rehab[0] ?? Object.keys(t)[0]) ?? "");
    }).catch(() => {});
    loadLabelEfficiency().then(setLabelEff).catch(() => setLabelEff(null));
    loadTransferMatrix().then(setLoco).catch(() => setLoco(null));
  }, []);

  const leChart = useMemo(
    () => (labelEff ?? []).map((p) => ({
      pct: +(p.fraction * 100).toFixed(1),
      miou: +(p.miou_mean ?? p.miou ?? 0).toFixed(3),
    })),
    [labelEff],
  );
  const trajKeys = Object.keys(traj);
  const series = traj[site] ?? [];
  const hasRecovery = series.some((p) => p.recovery_fraction != null);

  return (
    <div className="app">
      <header className="hero hero-3d">
        <div className="hero-text">
          <h1>Weak-supervised revegetation monitoring for mine rehabilitation</h1>
          <p className="sub">
            Vegetation classes (bare, water, senescent, vegetation) are mapped on rehabilitating
            Australian mine sites from Sentinel-2 imagery using weak, self-supervised labels. The
            study reports label efficiency, cross-site transfer, multi-temporal recovery, a
            fine-scale structure comparison, and per-pixel uncertainty.
          </p>
          <p className="byline">Sentinel-2 surface reflectance · SegFormer-B0 · leave-one-site-out cross-validation. Work in progress.</p>
        </div>
        <div>
          <SiteMap />
          <div className="map-legend"><span className="dot" /> rehabilitating mine sites (Sentinel-2 time series)</div>
        </div>
      </header>

      <Section id="seg" title="Semantic segmentation from weak labels">
        <p className="lead">
          A SegFormer-B0 model (RGB + NIR) is trained on CHM- and index-derived pseudo-labels rather
          than hand annotations. Per-pixel uncertainty is estimated with MC-dropout; it is highest at
          class boundaries, indicating where field verification would be most useful.
        </p>
        <UncertaintyGallery />
        <p className="cap">Held-out tiles. Left to right: Sentinel-2 RGB, predicted class map, MC-dropout uncertainty.</p>
        {val && (
          <>
            <p className="lead" style={{ marginTop: 18 }}>
              <strong>External validation.</strong> Because no manually labelled mine-rehab data exists,
              the weak labels are checked against {val.reference}, an independent product, over the
              {" "}{val.scheme} scheme. Pooled across {Object.keys(val.per_site).length} sites
              ({(val.pooled.n_pixels / 1e6).toFixed(1)}M pixels): Cohen's
              {" "}&kappa; = {val.pooled.kappa.toFixed(2)} ({(val.pooled.overall_agreement * 100).toFixed(0)}% agreement).
            </p>
            <table className="data">
              <caption>Agreement of weak labels with {val.reference} ({val.scheme}).</caption>
              <thead><tr><th>Site</th><th>Cohen's κ</th><th>Overall agreement</th><th>Pixels</th></tr></thead>
              <tbody>
                {Object.entries(val.per_site).map(([k, s]) => (
                  <tr key={k}>
                    <td>{prettyName(k)}</td>
                    <td>{s.kappa.toFixed(3)}</td>
                    <td>{(s.overall_agreement * 100).toFixed(1)}%</td>
                    <td>{s.n_pixels.toLocaleString()}</td>
                  </tr>
                ))}
                <tr style={{ fontWeight: 600 }}>
                  <td>Pooled</td>
                  <td>{val.pooled.kappa.toFixed(3)}</td>
                  <td>{(val.pooled.overall_agreement * 100).toFixed(1)}%</td>
                  <td>{val.pooled.n_pixels.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </>
        )}
      </Section>

      <Section id="labeleff" title="Label efficiency">
        <p className="lead">Validation mIoU as a function of the fraction of labelled tiles used in training.</p>
        {labelEff && labelEff.length > 0 && <LabelEffTable pts={labelEff} />}
        {leChart.length > 0 && (
          <div className="chart">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={leChart} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid stroke={GRID} />
                <XAxis dataKey="pct" type="number" scale="log" domain={["auto", "auto"]} stroke={AXIS}
                  label={{ value: "labelled fraction (%)", position: "insideBottom", offset: -4, fill: AXIS }} />
                <YAxis domain={[0, 1]} stroke={AXIS} label={{ value: "mIoU", angle: -90, position: "insideLeft", fill: AXIS }} />
                <Tooltip contentStyle={TIP} />
                <Line type="monotone" dataKey="miou" stroke={GREEN} strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {self && self.rows.length > 0 && (
          <>
            <p className="lead" style={{ marginTop: 18 }}>
              <strong>Self-training.</strong> At small label budgets the unused tiles are treated as
              unlabelled; confidence-thresholded (≥ {self.conf_threshold}) pseudo-labels from the
              supervised model are added and the model retrained. The benefit is largest when labels
              are scarcest and saturates as labels increase.
            </p>
            <table className="data">
              <caption>Supervised-only vs. self-training (validation mIoU).</caption>
              <thead>
                <tr><th>Labelled fraction</th><th>Labelled tiles</th><th>Supervised</th><th>Self-training</th><th>Δ</th></tr>
              </thead>
              <tbody>
                {self.rows.map((r) => (
                  <tr key={r.fraction}>
                    <td>{(r.fraction * 100).toFixed(0)}%</td>
                    <td>{r.n_labeled}</td>
                    <td>{r.miou_supervised.toFixed(3)}</td>
                    <td>{r.miou_self_training.toFixed(3)}</td>
                    <td style={{ color: r.gain > 0 ? GREEN : "#999" }}>
                      {r.gain >= 0 ? "+" : ""}{r.gain.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </Section>

      <Section id="trajectory" title="Restoration trajectory">
        <p className="lead">
          Annual mean NDVI and a standardised recovery fraction over each site's baseline-cleared
          footprint (2019&ndash;2024). Pixels are selected by their 2019 state only, so the trend is
          not selection-on-outcome.
        </p>
        <BeforeAfter />
        <p className="cap">Same ground in 2019 vs. 2024 over the strongest-recovery window at Alcoa
          Huntly — drag to compare (true colour or NDVI).</p>
        {trajKeys.length > 0 && (
          <label className="select">
            site:&nbsp;
            <select value={site} onChange={(e) => setSite(e.target.value)}>
              {trajKeys.map((k) => (
                <option key={k} value={k}>{prettySite(k)}{k.includes("rehab") ? " (cleared footprint)" : ""}</option>
              ))}
            </select>
          </label>
        )}
        <div className="chart">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={series} margin={{ top: 8, right: 32, left: 0, bottom: 8 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="year" stroke={AXIS} />
              <YAxis yAxisId="l" stroke={GREEN} domain={["auto", "auto"]}
                label={{ value: "mean NDVI", angle: -90, position: "insideLeft", fill: GREEN }} />
              <YAxis yAxisId="r" orientation="right" stroke={OCHRE} domain={[0, 1]} />
              <Tooltip contentStyle={TIP} />
              <Legend />
              <Line yAxisId="l" type="monotone" dataKey="mean_ndvi" name="mean NDVI" stroke={GREEN} strokeWidth={2} dot={{ r: 3 }} />
              {hasRecovery && (
                <Line yAxisId="r" type="monotone" dataKey="recovery_fraction" name="recovery fraction" stroke={OCHRE} strokeWidth={2} dot={{ r: 3 }} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section id="loco" title="Cross-site transfer">
        <p className="lead">
          A model trained on one mine and evaluated on the others. Within-site mIoU (diagonal) is
          generally higher than cross-site, which quantifies the domain gap of site-specific monitoring.
        </p>
        {loco && <LocoTable tm={loco} />}
      </Section>

      <Section id="bridge" title="Fine-scale structure vs. greenness">
        <p className="lead">
          A 1&nbsp;m canopy-height layer is aggregated to the 10&nbsp;m Sentinel-2 grid. NDVI saturates
          across the full range of canopy cover, so a green pixel may be tall forest or low regrowth;
          the fine-scale layer resolves the structure that the satellite alone cannot.
        </p>
        <p className="cap">
          Interactive 3D canopy-height surface (drag to orbit), sampled at the mine&ndash;forest
          boundary. Alcoa mines bauxite beneath jarrah forest, clearing in patches and progressively
          rehabilitating, so the terrain is a mosaic of cleared / regrowing ground and intact forest.
        </p>
        <Canopy3D />
        <div className="grid3">
          <figure className="fig"><img src={import.meta.env.BASE_URL + "assets/alcoa_huntly_bridge_chm_1m.png"} alt="1 m canopy height" /><figcaption>1 m canopy height</figcaption></figure>
          <figure className="fig"><img src={import.meta.env.BASE_URL + "assets/alcoa_huntly_bridge_canopyfrac_10m.png"} alt="canopy cover at 10 m" /><figcaption>canopy cover, aggregated to 10 m</figcaption></figure>
          <figure className="fig"><img src={import.meta.env.BASE_URL + "assets/alcoa_huntly_bridge_s2_ndvi_10m.png"} alt="Sentinel-2 NDVI 10 m" /><figcaption>Sentinel-2 NDVI, 10 m</figcaption></figure>
        </div>
      </Section>

      <footer className="foot">
        <p><strong>Guanxiong Huang</strong>, Northwest A&amp;F University ·{" "}
          <a href="mailto:harry.huang@nwafu.edu.cn">harry.huang@nwafu.edu.cn</a> ·{" "}
          <a href="https://github.com/Harry33t/mine-revegetation-rs">source code</a></p>
        <p>Open data: Sentinel-2 surface reflectance (Microsoft Planetary Computer), Meta global
          canopy height, ESA WorldCover. Methods: SegFormer-B0, self-training, MC-dropout
          uncertainty, leave-one-site-out cross-validation. Work in progress.</p>
      </footer>
    </div>
  );
}
