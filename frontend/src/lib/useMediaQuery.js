// Generic matchMedia-backed hook. App.jsx uses this to know whether the
// >=1180px "wide" layout is active, which decides where EvidencePanel mounts
// (frontend-design-plan.md §3.3 / §4.4 / §4.9): a single "drawer"-variant
// instance in AppShell's third grid column when wide, or an "inline"-variant
// instance rendered by whichever row/component owns the currently-open
// evidence source when narrow. EvidencePanel itself stays position-agnostic
// (plan §4.4) — this hook is what lets the caller make that decision.
//
// SSR-safe: falls back to `false` when `window.matchMedia` isn't available,
// and never throws.
import { useEffect, useState } from "react";

function getMatch(query) {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(query).matches
    : false;
}

export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => getMatch(query));

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return undefined;
    const mq = window.matchMedia(query);
    const handleChange = (event) => setMatches(event.matches);
    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, [query]);

  return matches;
}
