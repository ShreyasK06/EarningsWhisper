// The confidence meter for a SignalCard. Replaces the old ASCII "▓░" bar
// (defect 16) with a precise 20-tick track, an accessible role="meter", and
// a GPU-friendly mount animation. See frontend-design-plan.md §4.3.

import { clamp01 } from "../lib/format";

const LABELS = {
  bullish: "bullish",
  bearish: "bearish",
  neutral: "flat",
};

export default function ConfidenceMeter({ value, direction }) {
  const safeValue = clamp01(value);
  const dir = direction === "bullish" || direction === "bearish" ? direction : "neutral";
  const label = LABELS[dir];

  return (
    <div className={`confidence-meter confidence-meter--${dir}`}>
      <div
        className="confidence-meter__track"
        role="meter"
        aria-valuemin="0"
        aria-valuemax="1"
        aria-valuenow={safeValue}
        aria-valuetext={`${safeValue.toFixed(2)} confidence, ${label}`}
      >
        <div className="confidence-meter__ticks" aria-hidden="true" />
        <div className="confidence-meter__fill" style={{ "--conf": safeValue }} />
      </div>
      <div className="confidence-meter__legend t-micro" aria-hidden="true">
        <span>0.00</span>
        <span>0.50</span>
        <span>1.00</span>
      </div>
    </div>
  );
}
