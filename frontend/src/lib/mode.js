// Mirrors DIRECTIONAL_TRIGGERS and the keyword fast-path in api/main.py (lines 47-57).
// Keep this list byte-identical to the backend's list — it exists so the
// composer's "expecting a signal" hint matches the pipeline the server will
// actually run for the obvious cases. It is a prediction only: the backend's
// LLM-based router can still override it for ambiguous messages, and the
// resolved transcript entry is always labeled from the actual response type,
// never from this prediction.
export const DIRECTIONAL_TRIGGERS = [
  "going to go up", "going to go down", "will it go up",
  "should i buy", "should i sell", "bullish", "bearish",
  "outlook", "signal", "prediction", "worth buying",
];

/**
 * Predicts which pipeline a message will likely hit, mirroring the backend's
 * cheap keyword pass. Returns "signal" if any trigger substring is present
 * (case-insensitive), otherwise "lookup".
 */
export function predictMode(text) {
  const lowered = String(text ?? "").toLowerCase();
  return DIRECTIONAL_TRIGGERS.some((t) => lowered.includes(t)) ? "signal" : "lookup";
}
