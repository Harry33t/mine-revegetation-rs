import { useState } from "react";
import { asset } from "./data";

// MC-dropout triptychs exported by the trainer (uncertainty_0..5.png).
const N = 6;

export default function UncertaintyGallery() {
  const [i, setI] = useState(0);
  const imgs = Array.from({ length: N }, (_, k) => `uncertainty_${k}.png`);
  return (
    <div className="gallery">
      <img className="gallery-main" src={asset(imgs[i])} alt={`uncertainty sample ${i}`} />
      <div className="thumbs">
        {imgs.map((name, k) => (
          <button
            key={name}
            className={"thumb" + (k === i ? " active" : "")}
            onClick={() => setI(k)}
            aria-label={`sample ${k + 1}`}
          >
            <img src={asset(name)} alt={`thumb ${k}`} />
          </button>
        ))}
      </div>
    </div>
  );
}
