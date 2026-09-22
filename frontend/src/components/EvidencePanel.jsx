// The evidence surface — the product's proof, made a first-class panel
// instead of a native-tooltip `title` attribute (fixes defect 17). See
// frontend-design-plan.md §4.4.
//
// `variant`: "drawer" | "inline" (default "drawer"). This component is
// intentionally position-agnostic — it does not decide where it sits in the
// layout, only how its own content presents:
//   "drawer" — meant for the >=1180px third grid column: full padding,
//     the drawer drop-shadow, content slides in from the right.
//   "inline" — meant for the <1180px in-place expansion beneath a clicked
//     row: tighter padding, a left accent rule instead of a shadow, content
//     fades in from slightly above (no shadow — it isn't floating over the
//     transcript, it's part of it).
// The CALLER (App.jsx / AppShell.jsx, and whichever component tracks "which
// row is expanded" below 1180px) decides which variant to render and where
// to mount it; this component does not read viewport width itself.
//
// Escape-to-close and returning focus to the invoking row button are noted
// in the plan as App.jsx's responsibility (it owns the `evidence` state and
// the single onOpenEvidence/onClose wiring) — this component only moves
// focus to its own close button on open, since that side effect is fully
// local and can't conflict with a parent-level key handler.

import { useEffect, useRef } from "react";
import Icon from "./Icon";

export default function EvidencePanel({ source, onClose, variant = "drawer" }) {
  const closeRef = useRef(null);

  useEffect(() => {
    if (source) {
      closeRef.current?.focus();
    }
  }, [source]);

  if (!source) return null;

  const { chunkId, section, text, origin } = source;

  // The two honest "no excerpt" notices are both real answers to "why is
  // there no text": when this chunk id resolved to a known section, /chat's
  // answer shape simply never carries excerpt text; when section is also
  // null, the id wasn't found in the retrieved set at all. Never fabricate
  // an excerpt in either case.
  const notice =
    text != null
      ? null
      : section != null
        ? "The /chat answer response returns passage ids and sections only — the excerpt text isn't included for grounded lookups."
        : "The model cited this passage id, but it wasn't in the retrieved set returned by the API.";

  return (
    <aside
      className={`evidence-panel evidence-panel--${variant}`}
      role="complementary"
      aria-label="Cited passage"
    >
      <div className="evidence-panel__header">
        <div className="evidence-panel__heading">
          <p className="evidence-panel__chunk t-data">{chunkId ?? "—"}</p>
          <p className="evidence-panel__section t-micro">{section ?? "SECTION UNKNOWN"}</p>
        </div>
        <button
          type="button"
          ref={closeRef}
          className="evidence-panel__close"
          onClick={onClose}
          aria-label="Close cited passage"
        >
          <Icon name="close" size={14} />
        </button>
      </div>

      <div className="evidence-panel__body">
        {text != null ? (
          <blockquote className="evidence-panel__quote t-body-sm">
            <Icon name="quote" size={28} className="evidence-panel__glyph" aria-hidden="true" />
            {text}
          </blockquote>
        ) : (
          <div className="evidence-panel__notice">
            <p className="t-body-sm">{notice}</p>
          </div>
        )}
      </div>

      <p className="evidence-panel__footer t-micro">
        FILING EXCERPT · RETRIEVED PASSAGE
        {origin ? ` · ${origin.toUpperCase()}` : ""}
      </p>
    </aside>
  );
}
