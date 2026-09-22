// A brief terminal "boot" intro shown once per page load, purely as a
// signature first-impression moment for the research-terminal identity —
// it never blocks or delays the real /tickers fetch, which App.jsx kicks
// off in parallel underneath. Skipped entirely under prefers-reduced-motion.
import { useEffect, useState } from "react";

const LINES = ["EARNINGS WHISPER", "RESEARCH TERMINAL v0.3", "LOADING COVERAGE FEED..."];

function prefersReducedMotion() {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

export default function BootSequence({ onDone }) {
  const [phase, setPhase] = useState(() => (prefersReducedMotion() ? "done" : "in"));

  useEffect(() => {
    if (phase === "done") {
      onDone();
      return undefined;
    }
    const toOut = setTimeout(() => setPhase("out"), 950);
    return () => clearTimeout(toOut);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  useEffect(() => {
    if (phase !== "out") return undefined;
    const toDone = setTimeout(() => setPhase("done"), 360);
    return () => clearTimeout(toDone);
  }, [phase]);

  if (phase === "done") return null;

  return (
    <div className={`boot boot--${phase}`} aria-hidden="true">
      <div className="boot__lines">
        {LINES.map((line, i) => (
          <div className="boot__line t-data" style={{ "--i": i }} key={line}>
            <span className="boot__caret">&gt;</span> {line}
          </div>
        ))}
      </div>
    </div>
  );
}
