// The composed masthead shown in place of the transcript before the first
// message (plan §4.5). Replaces the old one-line empty state. Teaches both
// response modes up front via two prompt groups whose directional copy
// deliberately contains literal DIRECTIONAL_TRIGGERS substrings (see
// src/lib/mode.js) so they hit the backend's cheap keyword fast path.

import { companyName, dateline } from "../lib/format.js";
import { useSpotlight } from "../lib/useSpotlight.js";
import Icon from "./Icon.jsx";

// Computed once at module load, not per render — the dateline only needs
// day-level precision and this keeps LandingPanel a pure render function.
const TODAY = dateline(Date.now());

const LOOKUP_PROMPTS = [
  "What did management say about margins?",
  "Summarize the guidance in the latest filing.",
  "What risks were called out this quarter?",
];

function directionalPrompts(ticker) {
  return [
    `Is ${ticker} worth buying after this print?`,
    "What's the outlook from this quarter?",
    `Give me a signal on ${ticker}.`,
  ];
}

function PromptRow({ text, disabled, tag, onUse, i }) {
  const { onMouseMove } = useSpotlight();
  return (
    <button
      type="button"
      className={`landing__prompt-row landing__prompt-row--animate spotlight${tag ? " landing__prompt-row--directional" : ""}`}
      style={{ "--i": i }}
      aria-disabled={disabled ? "true" : undefined}
      onMouseMove={disabled ? undefined : onMouseMove}
      onClick={() => {
        if (!disabled) onUse(text);
      }}
    >
      <span className="landing__prompt-caret" aria-hidden="true">
        &gt;
      </span>
      <span className="landing__prompt-text t-ui">{text}</span>
      <span className="landing__prompt-trail">
        {tag && <span className="landing__prompt-tag t-micro">{tag}</span>}
        <Icon name="arrow-up-right" size={14} className="landing__prompt-arrow" />
      </span>
    </button>
  );
}

export default function LandingPanel({ ticker, tickers, onUsePrompt, offline, onRetryConnect }) {
  const directional = directionalPrompts(ticker);
  let i = 0;
  const nextIndex = () => i++;

  return (
    <div className="landing">
      <div className="landing__dateline t-meta landing__animate" style={{ "--i": nextIndex() }}>
        EARNINGS WHISPER · RESEARCH TERMINAL · {TODAY}
      </div>

      <div className="landing__masthead landing__animate" style={{ "--i": nextIndex() }}>
        <div className="landing__ticker">{ticker}</div>
        <div className="landing__company t-h1">{companyName(ticker)}</div>
      </div>

      <div className="landing__facts landing__animate" style={{ "--i": nextIndex() }}>
        <div className="landing__fact">
          <span className="landing__fact-label t-micro">
            <span className="landing__fact-dot landing__fact-dot--accent" aria-hidden="true" />
            COVERAGE
          </span>
          <span className="landing__fact-value t-data">
            {tickers.length} tickers
            {offline && <span className="landing__cached-tag t-micro">CACHED</span>}
          </span>
        </div>
        <div className="landing__fact">
          <span className="landing__fact-label t-micro">
            <span className="landing__fact-dot landing__fact-dot--teal" aria-hidden="true" />
            RETRIEVAL
          </span>
          <span className="landing__fact-value t-data">dense + hybrid</span>
        </div>
        <div className="landing__fact">
          <span className="landing__fact-label t-micro">
            <span className="landing__fact-dot landing__fact-dot--amber" aria-hidden="true" />
            OUTPUT
          </span>
          <span className="landing__fact-value t-data">citations required</span>
        </div>
      </div>

      {offline && (
        <div className="landing__offline-notice landing__animate" style={{ "--i": nextIndex() }}>
          <p className="t-body-sm">
            The API at localhost:8000 isn&apos;t responding. Start it with{" "}
            <code>uvicorn api.main:app --reload</code>, then retry.
          </p>
          <button type="button" className="landing__retry t-ui" onClick={onRetryConnect}>
            Retry connection
          </button>
        </div>
      )}

      <div className="landing__prompts">
        <div className="landing__prompt-group landing__animate" style={{ "--i": nextIndex() }}>
          <div className="landing__group-heading t-meta landing__group-heading--lookup">
            GROUNDED LOOKUP
          </div>
          <div className="landing__group-sub t-micro">
            dense retrieval · returns a cited answer
          </div>
          <div className="landing__rows">
            {LOOKUP_PROMPTS.map((text) => (
              <PromptRow
                key={text}
                text={text}
                disabled={offline}
                onUse={onUsePrompt}
                i={nextIndex()}
              />
            ))}
          </div>
        </div>

        <div className="landing__prompt-group landing__animate" style={{ "--i": nextIndex() }}>
          <div className="landing__group-heading t-meta landing__group-heading--directional">
            DIRECTIONAL CALL
          </div>
          <div className="landing__group-sub t-micro">
            hybrid retrieval + structured generation · slower
          </div>
          <div className="landing__rows">
            {directional.map((text) => (
              <PromptRow
                key={text}
                text={text}
                disabled={offline}
                tag="~10S"
                onUse={onUsePrompt}
                i={nextIndex()}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
