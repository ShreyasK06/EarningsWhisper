// Left instrument rail: coverage, live session stats, backend status.
// Replaces the old <select> ticker control (plan §4.1) — every ticker is
// now a real, accessibly-named <button>. Carries only data that is real:
// there is no persistence backend, so no fabricated "recent signals" or
// session history ever appears here (plan §3.1).

import { companyName } from "../lib/format.js";
import { useCountUp } from "../lib/useCountUp.js";

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
  sessionId,
  stats,
  onRetryConnect,
  tickersAreCached,
}) {
  const shortId = sessionId ? sessionId.slice(0, 8) : "—";
  const entriesDisplay = useCountUp(stats.entries, { duration: 500 });
  const lookupsDisplay = useCountUp(stats.lookups, { duration: 500 });
  const signalsDisplay = useCountUp(stats.signals, { duration: 500 });

  return (
    <aside className="rail" aria-label="Coverage and session">
      <div className="rail__wordmark">
        <div className="rail__brand t-h2">Earnings Whisper</div>
        <div className="rail__tag t-micro">RESEARCH TERMINAL · V0.3</div>
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

      <div className="rail__section rail__section--session">
        <span className="rail__label t-meta">SESSION</span>
        <div className="rail__stats">
          <div className="rail__stat-row">
            <span className="rail__stat-label t-micro">ID</span>
            <span className="rail__stat-value t-data">{shortId}</span>
          </div>
          <div className="rail__stat-row">
            <span className="rail__stat-label t-micro">ENTRIES</span>
            <span className="rail__stat-value t-data">{entriesDisplay}</span>
          </div>
          <div className="rail__stat-row">
            <span className="rail__stat-label t-micro">LOOKUPS</span>
            <span className="rail__stat-value t-data">{lookupsDisplay}</span>
          </div>
          <div className="rail__stat-row">
            <span className="rail__stat-label t-micro">SIGNALS</span>
            <span className="rail__stat-value t-data">{signalsDisplay}</span>
          </div>
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
