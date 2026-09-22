// The loading state, differentiated by expected mode (fixes defect 24 — a
// single 3-dot spinner served both a ~1-3s lookup and an ~8-25s signal
// generation identically). See frontend-design-plan.md §4.2.
//
// Timers are keyed off `entry.startedAt` (wall-clock), not off mount time
// via setTimeout chains, so the elapsed display and staged checklist stay
// correct even if the component remounts.

import { useEffect, useState } from "react";

const SIGNAL_STEPS = [
  { key: "retrieval", label: "hybrid retrieval · 4 queries", at: 0 },
  { key: "ranking", label: "ranking passages", at: 1400 },
  { key: "generation", label: "structured generation", at: 3200 },
  { key: "validating", label: "validating citations", at: 7000 },
];

const REASSURANCE_AT_MS = 12000;
const TICK_MS = 1000;

function useReducedMotion() {
  const getMatch = () =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

  const [reduced, setReduced] = useState(getMatch);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handleChange = (event) => setReduced(event.matches);
    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, []);

  return reduced;
}

// `startedAt` must already be a resolved timestamp by the time it reaches
// this hook (see the lazy useState fallback in PendingEntry below) — this
// keeps the only impure Date.now() calls inside lazy initializers/timer
// callbacks, never in the render body itself.
function useElapsedMs(startedAt) {
  const [elapsed, setElapsed] = useState(() => Date.now() - startedAt);

  useEffect(() => {
    const id = setInterval(() => setElapsed(Date.now() - startedAt), TICK_MS);
    return () => clearInterval(id);
  }, [startedAt]);

  return elapsed;
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function LookupPending({ elapsedMs, reducedMotion, showReassurance }) {
  return (
    <div className="pending pending--lookup">
      <p className="pending__status t-meta">
        {reducedMotion ? "WORKING…" : "RETRIEVING PASSAGES"}
      </p>
      {!reducedMotion && (
        <div className="pending__skeleton" aria-hidden="true">
          <span className="pending__bar" style={{ width: "92%" }} />
          <span className="pending__bar" style={{ width: "86%" }} />
          <span className="pending__bar" style={{ width: "54%" }} />
        </div>
      )}
      <p className="pending__timer t-data">{formatElapsed(elapsedMs)}</p>
      {showReassurance && (
        <p className="pending__reassurance t-micro">
          still working — the model is slower than usual
        </p>
      )}
    </div>
  );
}

function SignalPending({ elapsedMs, reducedMotion, showReassurance }) {
  const activeIndex = SIGNAL_STEPS.reduce(
    (acc, step, i) => (elapsedMs >= step.at ? i : acc),
    0,
  );

  return (
    <div className="pending pending--signal">
      <p className="pending__status t-meta">
        {reducedMotion ? "WORKING…" : "GENERATING SIGNAL"}
      </p>
      <ul className="pending__checklist">
        {SIGNAL_STEPS.map((step, i) => {
          const state = i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          return (
            <li key={step.key} className={`pending__step pending__step--${state}`}>
              <span
                className={`pending__step-marker pending__step-marker--${state}`}
                aria-hidden="true"
              />
              <span className="pending__step-label t-ui">{step.label}</span>
            </li>
          );
        })}
      </ul>
      <p className="pending__timer t-data">{formatElapsed(elapsedMs)}</p>
      {showReassurance && (
        <p className="pending__reassurance t-micro">
          still working — signal generation runs a full hybrid sweep
        </p>
      )}
    </div>
  );
}

export default function PendingEntry({ entry }) {
  // Lazy initializer: Date.now() only runs (once) as a last-resort fallback
  // for a malformed entry missing both `startedAt` and `at`.
  const [startedAt] = useState(() => entry?.startedAt ?? entry?.at ?? Date.now());
  const expectedMode = entry?.expectedMode === "signal" ? "signal" : "lookup";
  const reducedMotion = useReducedMotion();
  const elapsedMs = useElapsedMs(startedAt);
  const showReassurance = elapsedMs >= REASSURANCE_AT_MS;

  if (expectedMode === "signal") {
    return (
      <SignalPending
        elapsedMs={elapsedMs}
        reducedMotion={reducedMotion}
        showReassurance={showReassurance}
      />
    );
  }

  return (
    <LookupPending
      elapsedMs={elapsedMs}
      reducedMotion={reducedMotion}
      showReassurance={showReassurance}
    />
  );
}
