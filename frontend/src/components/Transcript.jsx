// The research-log transcript. Replaces ChatWindow — see
// frontend-design-plan.md §4.2. Renders the app's <main id="transcript">
// landmark itself; if a future App/AppShell rewrite wraps its own <main>,
// that owner is responsible for not nesting a second one.
//
// `entries` is expected to be non-empty in normal operation (the empty/
// landing state is a sibling component owned by App.jsx), but this still
// renders null defensively rather than assuming a non-empty array.

import { useEffect, useRef } from "react";
import TranscriptEntry from "./TranscriptEntry";

const AUTO_SCROLL_THRESHOLD_PX = 120;

export default function Transcript({
  entries,
  ticker,
  onRetry,
  onOpenEvidence,
  evidence,
  isWideViewport,
  onCloseEvidence,
}) {
  const containerRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom > AUTO_SCROLL_THRESHOLD_PX) return;

    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries]);

  if (!Array.isArray(entries) || entries.length === 0) {
    return null;
  }

  return (
    <main
      id="transcript"
      className="transcript"
      role="log"
      aria-live="off"
      ref={containerRef}
    >
      {entries.map((entry, index) => (
        <TranscriptEntry
          key={entry.id}
          entry={entry}
          index={index}
          ticker={ticker}
          onRetry={onRetry}
          onOpenEvidence={onOpenEvidence}
          evidence={evidence}
          isWideViewport={isWideViewport}
          onCloseEvidence={onCloseEvidence}
        />
      ))}
      <div className="transcript__end" ref={bottomRef} aria-hidden="true" />
    </main>
  );
}
