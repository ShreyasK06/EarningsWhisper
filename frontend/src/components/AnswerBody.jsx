// Renders a `payload.type === "answer"` entry's prose plus its Sources
// block. See frontend-design-plan.md §4.2. Fixes defect 19 (paragraphs +
// inline [chunk_id] markers instead of a raw-bracketed wall of text) and
// defect 20 (distinguishing "cited inline" from "retrieved only" sources —
// /chat's citations[] is every dense_search hit, not only what the model
// used).

import { parseInlineRefs, buildSourceList } from "../lib/citations";
import EvidenceRow from "./EvidenceRow";
import EvidencePanel from "./EvidencePanel";

export default function AnswerBody({
  payload,
  onOpenEvidence,
  evidence,
  isWideViewport,
  onCloseEvidence,
}) {
  const text = typeof payload?.text === "string" ? payload.text : "";
  const citations = Array.isArray(payload?.citations) ? payload.citations : [];
  const { paragraphs, refs } = parseInlineRefs(text);
  const sources = buildSourceList(citations, refs);

  function sectionForChunk(chunkId) {
    const match = citations.find((c) => (c?.chunk_id ?? c?.chunkId) === chunkId);
    return match?.section ?? null;
  }

  function openChunk(chunkId, section) {
    onOpenEvidence?.({ chunkId, section, text: null, origin: "answer" });
  }

  // Below 1180px there is no drawer column — EvidencePanel is rendered
  // inline by whichever row owns the open source instead (plan §3.3/§4.4).
  // Matched by chunkId + origin so an answer entry never lights up a row for
  // evidence opened from an unrelated signal card elsewhere in the
  // transcript. Only one row can match at a time, so only one panel mounts.
  const isNarrowAndOpenHere = (chunkId) =>
    !isWideViewport && evidence != null && evidence.origin === "answer" && evidence.chunkId === chunkId;

  // A chunk id can be cited inline but absent from payload.citations (the
  // model referenced a passage id the API didn't return) — there is no
  // Sources row for it to expand under, so it has no home in the sources
  // list at all. Render it just above that list instead of silently
  // dropping the click on narrow viewports.
  const hasMatchingSource = sources.some((s) => s.chunkId === evidence?.chunkId);
  const showOrphanInline =
    !isWideViewport && evidence != null && evidence.origin === "answer" && !hasMatchingSource;

  return (
    <div className="answer-body">
      {paragraphs.length === 0 ? (
        <p className="answer-body__paragraph t-body">—</p>
      ) : (
        paragraphs.map((tokens, pIndex) => (
          <p className="answer-body__paragraph t-body" key={pIndex}>
            {tokens.map((token, tIndex) =>
              token.type === "ref" ? (
                <button
                  key={tIndex}
                  type="button"
                  className="ref"
                  onClick={() => openChunk(token.chunkId, sectionForChunk(token.chunkId))}
                  aria-label={`Open cited passage ${token.number}`}
                >
                  {token.number}
                </button>
              ) : (
                <span key={tIndex}>{token.value}</span>
              ),
            )}
          </p>
        ))
      )}

      {showOrphanInline && (
        <EvidencePanel source={evidence} variant="inline" onClose={onCloseEvidence} />
      )}

      {sources.length > 0 && (
        <div className="answer-body__sources">
          <p className="answer-body__sources-label t-meta">SOURCES · {sources.length}</p>
          <ul className="answer-body__sources-list">
            {sources.map((source, i) => (
              <li
                key={`${source.chunkId}-${i}`}
                className={
                  source.citedInline
                    ? "answer-body__source"
                    : "answer-body__source answer-body__source--muted"
                }
              >
                <div className="answer-body__source-row">
                  <EvidenceRow
                    index={source.citedInline ? source.number : "·"}
                    section={source.section}
                    chunkId={source.chunkId}
                    preview={null}
                    onOpen={() => openChunk(source.chunkId, source.section)}
                  />
                  {!source.citedInline && (
                    <span className="answer-body__tag t-micro">RETRIEVED, NOT CITED</span>
                  )}
                </div>
                {isNarrowAndOpenHere(source.chunkId) && (
                  <EvidencePanel source={evidence} variant="inline" onClose={onCloseEvidence} />
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
