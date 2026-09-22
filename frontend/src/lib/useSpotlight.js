// Cursor-tracked spotlight highlight, paired with the .spotlight utility in
// base.css. Returns an onMouseMove handler that writes --mx/--my as
// percentages onto the event's currentTarget — no state, no re-render, so
// it's cheap enough to attach to a whole list of rows.
//
// Pass { tilt: true } to also drive a subtle 3D tilt via --rx/--ry (paired
// with the .tilt utility in base.css) — reserved for the one surface that
// wants that extra weight (SignalCard), not applied broadly.
export function useSpotlight({ tilt = false, maxTilt = 4 } = {}) {
  function onMouseMove(event) {
    const el = event.currentTarget;
    const rect = el.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * 100;
    const my = ((event.clientY - rect.top) / rect.height) * 100;
    el.style.setProperty("--mx", `${mx}%`);
    el.style.setProperty("--my", `${my}%`);

    if (tilt) {
      const nx = mx / 100 - 0.5; // -0.5..0.5
      const ny = my / 100 - 0.5;
      el.style.setProperty("--ry", `${(nx * maxTilt * 2).toFixed(2)}deg`);
      el.style.setProperty("--rx", `${(-ny * maxTilt * 2).toFixed(2)}deg`);
    }
  }

  function onMouseLeave(event) {
    if (!tilt) return;
    const el = event.currentTarget;
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  }

  return { onMouseMove, onMouseLeave };
}
