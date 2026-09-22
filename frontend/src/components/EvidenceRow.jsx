// The shared clickable evidence row. Used inline by AnswerBody's Sources
// block AND by SignalCard's Supporting Evidence block (the latter owned by
// a different task/agent — this file's prop contract is normative per
// frontend-design-plan.md §4.3/§4.4 and must not change shape).
//
// Props: { index, section, chunkId, preview, onOpen }
//   index    — number (rendered zero-padded, "01", "02", …) or a pre-
//              formatted string (e.g. a superscript ref number, or "·" for
//              "retrieved but not cited"). Rendered as-is when not a number.
//   section  — string | null. Falls back to "SECTION UNKNOWN" when absent,
//              never left blank (this is a readable-fallback requirement).
//   chunkId  — string. Right-aligned, mono, wraps anywhere so long ids never
//              force horizontal scroll.
//   preview  — optional one-line excerpt preview string. Omitted entirely
//              when falsy (e.g. AnswerBody has no excerpt text to preview).
//   onOpen   — click handler; the whole row is one <button>.

import { pad2 } from "../lib/format";

export default function EvidenceRow({ index, section, chunkId, preview, onOpen }) {
  const displayIndex = typeof index === "number" ? pad2(index) : (index ?? "·");

  return (
    <button type="button" className="evidence-row" onClick={onOpen}>
      <span className="evidence-row__index t-micro" aria-hidden="true">
        {displayIndex}
      </span>
      <span className="evidence-row__main">
        <span className="evidence-row__section t-ui">{section ?? "SECTION UNKNOWN"}</span>
        {preview ? <span className="evidence-row__preview t-body-sm">{preview}</span> : null}
      </span>
      <span className="evidence-row__chunk t-micro">{chunkId}</span>
    </button>
  );
}
