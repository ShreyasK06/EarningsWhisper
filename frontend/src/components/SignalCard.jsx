// Renders a raw `payload.signal` object as a directional-call research card.
// Rewrite per frontend-design-plan.md §4.3 — replaces the old top-bar card
// with an ASCII confidence bar (defects 16, 17, 18).

import { useState } from "react";
import ConfidenceMeter from "./ConfidenceMeter";
import EvidenceRow from "./EvidenceRow";
import EvidencePanel from "./EvidencePanel";
import { clamp01, clockTime, dateline, truncate } from "../lib/format";
import { useSpotlight } from "../lib/useSpotlight";
import { useCountUp } from "../lib/useCountUp";

const DIRECTION_LABELS = {
  bullish: "bullish",
  bearish: "bearish",
  neutral: "flat",
};

function normalizeDirection(direction) {
  return direction === "bullish" || direction === "bearish" ? direction : "neutral";
}

// Date.now() is correct here — a signal is generated live, so "now" is the
// right dateline. Guarded per the plan: some sandboxed contexts restrict
// Date.now(), and this must never crash the card.
function safeNow() {
  try {
    return Date.now();
  } catch {
    return null;
  }
}

export default function SignalCard({
  signal,
  onOpenEvidence,
  evidence,
  isWideViewport,
  onCloseEvidence,
}) {
  const [generatedAt] = useState(safeNow);
  const { onMouseMove, onMouseLeave } = useSpotlight({ tilt: true });
  const confidenceDisplay = useCountUp(clamp01(signal?.confidence), {
    duration: 750,
    decimals: 2,
  });

  const ticker = signal?.ticker ?? "—";
  const direction = normalizeDirection(signal?.signal);
  const label = DIRECTION_LABELS[direction];
  const confidence = clamp01(signal?.confidence);
  const reasoning = typeof signal?.reasoning === "string" ? signal.reasoning : "";
  const paragraphs = reasoning
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  const claims = Array.isArray(signal?.supporting_claims) ? signal.supporting_claims : [];

  return (
    <div
      className={`signal-card spotlight signal-card--${direction}`}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
    >
      <div className="signal-card__header">
        <div className="signal-card__header-left">
          <span className="signal-card__ticker t-data">{ticker}</span>
          <span className="signal-card__hairline" aria-hidden="true" />
          <span className="signal-card__kicker t-meta">DIRECTIONAL CALL</span>
        </div>
        <span className="signal-card__dateline t-micro">
          {dateline(generatedAt)} · {clockTime(generatedAt)}
        </span>
      </div>

      <div className="signal-card__verdict">
        <span className="signal-card__stamp">
          <span className="signal-card__verdict-word">{label}</span>
        </span>
        <div className="signal-card__confidence-figure">
          <span className="signal-card__confidence-value t-data-lg">
            {confidenceDisplay.toFixed(2)}
          </span>
          <span className="signal-card__confidence-label t-micro">CONFIDENCE</span>
        </div>
      </div>

      <ConfidenceMeter value={confidence} direction={direction} />

      <div className="signal-card__reasoning">
        {paragraphs.length > 0 ? (
          paragraphs.map((p, i) => <p key={i}>{p}</p>)
        ) : (
          <p>—</p>
        )}
      </div>

      <div className="signal-card__evidence">
        <span className="signal-card__evidence-label t-meta">
          SUPPORTING EVIDENCE · {claims.length}
        </span>
        {claims.length > 0 ? (
          <div className="signal-card__evidence-list">
            {claims.map((claim, i) => {
              const chunkId = claim?.chunk_id ?? "—";
              // Below 1180px there is no drawer column — the row that
              // opened this claim's evidence renders its own inline panel
              // instead (plan §3.3/§4.4). Matched by chunkId + origin so a
              // signal card never lights up a claim for evidence opened
              // from an unrelated answer entry elsewhere in the transcript.
              const openHere =
                !isWideViewport &&
                evidence != null &&
                evidence.origin === "signal" &&
                evidence.chunkId === chunkId;
              return (
                <div className="signal-card__evidence-item" key={chunkId !== "—" ? chunkId : i}>
                  <EvidenceRow
                    index={i + 1}
                    section={claim?.section ?? null}
                    chunkId={chunkId}
                    preview={truncate(claim?.text ?? "", 160)}
                    onOpen={() =>
                      onOpenEvidence?.({
                        chunkId: claim?.chunk_id,
                        section: claim?.section,
                        text: claim?.text,
                        origin: "signal",
                      })
                    }
                  />
                  {openHere && (
                    <EvidencePanel source={evidence} variant="inline" onClose={onCloseEvidence} />
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="signal-card__evidence-empty t-body-sm">—</p>
        )}
      </div>

      <div className="signal-card__footer t-micro">MODEL OUTPUT · NOT INVESTMENT ADVICE</div>
    </div>
  );
}
