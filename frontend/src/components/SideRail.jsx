// Left instrument rail: coverage, appearance controls, backend status.
// Replaces the old <select> ticker control (plan §4.1) — every ticker is
// now a real, accessibly-named <button>. Carries only data that is real:
// there is no persistence backend, so no fabricated "recent signals" or
// session history ever appears here (plan §3.1).

import { useState } from "react";
import { companyName } from "../lib/format.js";
import { ACCENTS, useTheme } from "../lib/useTheme.js";
import Icon from "./Icon.jsx";

const STATUS_COPY = {
  online: "API ONLINE",
  offline: "API OFFLINE",
  connecting: "CONNECTING…",
};

export default function SideRail({
  tickers,
  activeTicker,
  onSelectTicker,
  status,
  onRetryConnect,
  tickersAreCached,
}) {
  const { theme, toggleTheme, accent, setAccent } = useTheme();
  const [accentPickerOpen, setAccentPickerOpen] = useState(false);

  return (
    <aside className="rail" aria-label="Coverage and appearance">
      <div className="rail__wordmark">
        <div className="rail__brand t-h2">Earnings Whisper</div>
        <div className="rail__appearance">
          <button
            type="button"
            className="rail__appearance-btn"
            aria-pressed={theme === "dark"}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            onClick={toggleTheme}
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} size={14} />
          </button>
          <div className="rail__accent-picker">
            <button
              type="button"
              className="rail__appearance-btn"
              aria-expanded={accentPickerOpen}
              aria-label="Change accent color"
              onClick={() => setAccentPickerOpen((prev) => !prev)}
            >
              <Icon name="palette" size={14} />
            </button>
            {accentPickerOpen && (
              <div className="rail__accent-swatches" role="radiogroup" aria-label="Accent color">
                {ACCENTS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    role="radio"
                    aria-checked={accent === option.id}
                    aria-label={option.label}
                    className="rail__accent-swatch"
                    data-selected={accent === option.id || undefined}
                    style={{ "--swatch": option.swatch }}
                    onClick={() => {
                      setAccent(option.id);
                      setAccentPickerOpen(false);
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="rail__section rail__section--coverage">
        <div className="rail__label-row">
          <span className="rail__label t-meta">COVERAGE</span>
          {tickersAreCached && <span className="rail__tag-cached t-micro">CACHED</span>}
        </div>
        <div className="rail__tickers">
          {tickers.map((ticker, i) => {
            const selected = ticker === activeTicker;
            return (
              <button
                key={ticker}
                type="button"
                className="rail__ticker rail__ticker--animate"
                aria-pressed={selected}
                data-selected={selected || undefined}
                style={{ "--i": i }}
                onClick={() => onSelectTicker(ticker)}
              >
                <span className="rail__ticker-tag" aria-hidden="true" />
                <span className="rail__ticker-symbol t-data">{ticker}</span>
                <span className="rail__ticker-name t-micro">{companyName(ticker)}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="rail__status">
        <div className="rail__status-row">
          <span
            className={`rail__status-square rail__status-square--${status}`}
            aria-hidden="true"
          />
          <span className="rail__status-label t-micro">{STATUS_COPY[status]}</span>
        </div>
        <div className="rail__status-host t-micro">localhost:8000</div>
        {status === "offline" && (
          <button type="button" className="rail__retry t-micro" onClick={onRetryConnect}>
            Retry
          </button>
        )}
      </div>
    </aside>
  );
}
