// Parses `[chunk_id]` inline reference markers out of model-generated prose
// and builds the retrieved-vs-cited distinction for a citations list.
// See frontend-design-plan.md §4.2 (defect 19 and defect 20).

const REF_PATTERN = /\[([A-Za-z0-9_.:-]+)\]/g;

/**
 * Splits `text` into paragraphs on blank lines and extracts `[chunk_id]`
 * inline reference markers, assigning each distinct chunk id a sequential
 * number in order of first appearance across the whole text.
 *
 * Returns:
 *   paragraphs: Array<Array<{ type: "text", value: string } | { type: "ref", chunkId: string, number: number }>>
 *   refs: Map<chunkId, number>  — first-appearance order, 1-based
 *
 * Never throws: missing/empty/non-string input yields { paragraphs: [], refs: new Map() }.
 */
export function parseInlineRefs(text) {
  const raw = typeof text === "string" ? text : "";
  const refs = new Map();
  let nextNumber = 1;

  const rawParagraphs = raw
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const paragraphs = rawParagraphs.map((paragraph) => {
    const tokens = [];
    let lastIndex = 0;

    for (const match of paragraph.matchAll(REF_PATTERN)) {
      const chunkId = match[1];
      const start = match.index ?? 0;

      if (start > lastIndex) {
        tokens.push({ type: "text", value: paragraph.slice(lastIndex, start) });
      }

      if (!refs.has(chunkId)) {
        refs.set(chunkId, nextNumber++);
      }
      tokens.push({ type: "ref", chunkId, number: refs.get(chunkId) });

      lastIndex = start + match[0].length;
    }

    if (lastIndex < paragraph.length) {
      tokens.push({ type: "text", value: paragraph.slice(lastIndex) });
    }

    return tokens;
  });

  return { paragraphs, refs };
}

/**
 * Builds the evidence source list for an answer's Sources block, deduped by
 * chunk_id, marking each citation as "cited inline" (its chunk_id appears in
 * `refs`, i.e. the model actually referenced it in prose) vs "retrieved
 * only" (returned by dense_search but never cited). This distinction is the
 * fix for defect 20 — /chat's `citations[]` is every retrieved hit, not only
 * what the model used.
 *
 * `citations` — payload.citations, an array of { chunk_id, section }.
 * `refs` — a Map<chunkId, number> (or plain object) from parseInlineRefs.
 *
 * Returns Array<{ chunkId, section, citedInline: boolean, number: number | null }>
 * in the original citation order. Never throws on malformed input.
 */
export function buildSourceList(citations, refs) {
  const list = Array.isArray(citations) ? citations : [];
  const refMap = refs instanceof Map ? refs : new Map(Object.entries(refs ?? {}));

  const seen = new Set();
  const result = [];

  for (const citation of list) {
    if (!citation || typeof citation !== "object") continue;
    const chunkId = citation.chunk_id ?? citation.chunkId;
    if (chunkId == null || seen.has(chunkId)) continue;
    seen.add(chunkId);

    const number = refMap.has(chunkId) ? refMap.get(chunkId) : null;
    result.push({
      chunkId,
      section: citation.section ?? null,
      citedInline: number != null,
      number,
    });
  }

  return result;
}
