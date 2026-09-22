import { useEffect, useRef, useState } from "react";

function prefersReducedMotion() {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

// Animates a displayed number toward `target` whenever it changes — used for
// the confidence figure and the rail's live session stats, so a value
// landing feels like it arrived rather than snapping into place. Skips the
// tween entirely under prefers-reduced-motion.
export function useCountUp(target, { duration = 700, decimals = 0 } = {}) {
  const [display, setDisplay] = useState(target);
  const fromRef = useRef(target);
  const rafRef = useRef(null);

  useEffect(() => {
    const from = fromRef.current;
    if (from === target) return undefined;

    if (prefersReducedMotion()) {
      setDisplay(target);
      fromRef.current = target;
      return undefined;
    }

    const start = performance.now();
    function tick(now) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / duration);
      const eased = 1 - (1 - t) * (1 - t) * (1 - t); // ease-out cubic
      const value = from + (target - from) * eased;
      setDisplay(Number(value.toFixed(decimals)));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
      }
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);

  return display;
}
