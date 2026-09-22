// Auto-growing message composer. Replaces InputBar.jsx (plan §4.6) — same
// prop contract as the old component so App.jsx can swap the import in
// without further changes.

import { useRef, useState } from "react";
import Icon from "./Icon";
import { predictMode } from "../lib/mode";

const MAX_HEIGHT_REM = 9.5;

export default function Composer({ onSend, disabled, loading, ticker }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  const mode = predictMode(text);
  // Preserves the exact existing InputBar gating (!ticker || loading) in
  // addition to whatever `disabled` the parent passes, so the composer can
  // never be sent through while there is no ticker or a request in flight.
  const isDisabled = Boolean(disabled) || !ticker || loading;

  function resizeTextarea(el) {
    if (!el) return;
    el.style.height = "auto";
    const maxHeight = parseFloat(getComputedStyle(el).fontSize) * MAX_HEIGHT_REM;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight || el.scrollHeight)}px`;
  }

  function handleChange(e) {
    setText(e.target.value);
    resizeTextarea(e.target);
  }

  function resetHeight() {
    const el = textareaRef.current;
    if (el) el.style.height = "auto";
  }

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || isDisabled) return;
    onSend(trimmed);
    setText("");
    resetHeight();
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
    // Shift+Enter: default behavior inserts a newline — don't preventDefault.
  }

  function handleSubmit(e) {
    e.preventDefault();
    submit();
  }

  const placeholder = loading
    ? "Waiting on the model…"
    : `Ask about the ${ticker ?? "…"} filing…`;

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer__row">
        <div className="composer__field">
          <span className="composer__prefix" aria-hidden="true">
            &gt;
          </span>
          {!text && <span className="composer__cursor" aria-hidden="true" />}
          <textarea
            ref={textareaRef}
            className="composer__textarea"
            rows={1}
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isDisabled}
          />
        </div>
        <button
          type="submit"
          className="composer__send"
          aria-label="Send message"
          disabled={!text.trim() || isDisabled}
        >
          <Icon name="arrow-right" />
        </button>
      </div>

      <div className="composer__meta">
        <span className={`composer__mode-chip t-micro composer__mode-chip--${mode}`}>
          {mode === "signal" ? "EXPECTING A SIGNAL · ~10S" : "GROUNDED LOOKUP"}
        </span>
        <span className="composer__hint t-micro">
          ENTER TO SEND · SHIFT+ENTER FOR A NEW LINE
        </span>
      </div>
    </form>
  );
}
