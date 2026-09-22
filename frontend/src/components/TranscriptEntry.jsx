// Shared per-entry skeleton (gutter + body) and role/type routing. See
// frontend-design-plan.md §4.2. Branches on `entry.role`, and for
// `role: "assistant"` branches again on `payload.type` — an unrecognized
// type always renders a readable raw-JSON fallback, never a blank screen.

import { useState } from "react";
import { pad2, clockTime } from "../lib/format";
import AnswerBody from "./AnswerBody";
import PendingEntry from "./PendingEntry";
import FailureEntry from "./FailureEntry";
import SignalCard from "./SignalCard";

const DIRECTION_KIND_CLASS = {
  bullish: "entry__kind--signal-bullish",
  bearish: "entry__kind--signal-bearish",
  neutral: "entry__kind--signal-neutral",
};

function resolveKind(entry) {
  switch (entry.role) {
    case "user":
      return { label: "QUERY", className: "entry__kind--query" };
    case "assistant": {
      const type = entry.payload?.type;
      if (type === "answer") {
        return { label: "LOOKUP", className: "entry__kind--lookup" };
      }
      if (type === "signal") {
        const direction = entry.payload?.signal?.signal;
        return {
          label: "SIGNAL",
          className: DIRECTION_KIND_CLASS[direction] ?? "entry__kind--signal-neutral",
        };
      }
      return { label: "UNKNOWN", className: "entry__kind--unknown" };
    }
    case "pending":
      return { label: "WORKING", className: "entry__kind--working" };
    case "failure":
      return { label: "ERROR", className: "entry__kind--error" };
    default:
      return { label: "ENTRY", className: "entry__kind--unknown" };
  }
}

function UnknownFallback({ payload }) {
  let json;
  try {
    json = JSON.stringify(payload, null, 2);
  } catch {
    json = String(payload);
  }
  return (
    <div className="entry__unknown">
      <p className="entry__unknown-label t-ui">
        Unrecognized response shape — showing the raw payload.
      </p>
      <pre className="entry__unknown-json t-micro">{json}</pre>
    </div>
  );
}

// `evidence`/`isWideViewport`/`onCloseEvidence` flow all the way down to
// AnswerBody/SignalCard so their EvidenceRows can each answer "is the
// currently-open evidence source mine?" and render their own inline
// EvidencePanel when narrow (frontend-design-plan.md §3.3/§4.4/§4.9). At
// >=1180px these are effectively unused by the children — App.jsx renders
// the single drawer instance itself in that case.
function AssistantBody({ entry, onOpenEvidence, evidence, isWideViewport, onCloseEvidence }) {
  const payload = entry.payload;
  if (!payload || typeof payload !== "object") {
    return <UnknownFallback payload={payload} />;
  }
  if (payload.type === "answer") {
    return (
      <AnswerBody
        payload={payload}
        onOpenEvidence={onOpenEvidence}
        evidence={evidence}
        isWideViewport={isWideViewport}
        onCloseEvidence={onCloseEvidence}
      />
    );
  }
  if (payload.type === "signal") {
    return (
      <SignalCard
        signal={payload.signal}
        onOpenEvidence={onOpenEvidence}
        evidence={evidence}
        isWideViewport={isWideViewport}
        onCloseEvidence={onCloseEvidence}
      />
    );
  }
  return <UnknownFallback payload={payload} />;
}

function EntryBody({ entry, onRetry, onOpenEvidence, evidence, isWideViewport, onCloseEvidence }) {
  switch (entry.role) {
    case "user":
      return <div className="entry__query t-data">{entry.text}</div>;
    case "assistant":
      return (
        <AssistantBody
          entry={entry}
          onOpenEvidence={onOpenEvidence}
          evidence={evidence}
          isWideViewport={isWideViewport}
          onCloseEvidence={onCloseEvidence}
        />
      );
    case "pending":
      return <PendingEntry entry={entry} />;
    case "failure":
      return (
        <FailureEntry
          reason={entry.reason}
          message={entry.message}
          retryText={entry.retryText}
          onRetry={onRetry}
        />
      );
    default:
      return <UnknownFallback payload={entry} />;
  }
}

export default function TranscriptEntry({
  entry,
  index,
  onRetry,
  onOpenEvidence,
  evidence,
  isWideViewport,
  onCloseEvidence,
}) {
  // Called unconditionally (before the "notice" early return below) to
  // satisfy react-hooks/rules-of-hooks. Lazy initializer: Date.now() only
  // runs (once) as a last-resort fallback for a malformed entry missing
  // both `at` and `startedAt` — never as a live-updating value recomputed
  // on every render.
  const [timestamp] = useState(() => entry.at ?? entry.startedAt ?? Date.now());

  if (entry.role === "notice") {
    return (
      <div className="entry entry--notice">
        <span className="entry__notice-text t-micro">{entry.message}</span>
      </div>
    );
  }

  const kind = resolveKind(entry);
  let isoTime;
  try {
    isoTime = new Date(timestamp).toISOString();
  } catch {
    isoTime = undefined;
  }

  return (
    <article className={`entry entry--${entry.role}`}>
      <div className="entry__gutter">
        <span className="entry__index t-meta">{pad2(index + 1)}</span>
        <span className={`entry__kind t-meta ${kind.className}`}>{kind.label}</span>
        <time className="entry__time t-meta" dateTime={isoTime}>
          {clockTime(timestamp)}
        </time>
      </div>
      <div className="entry__body">
        <EntryBody
          entry={entry}
          onRetry={onRetry}
          onOpenEvidence={onOpenEvidence}
          evidence={evidence}
          isWideViewport={isWideViewport}
          onCloseEvidence={onCloseEvidence}
        />
      </div>
    </article>
  );
}
