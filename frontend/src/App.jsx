// App shell state and orchestration — frontend-design-plan.md §4.9.
//
// Owns: tickers/ticker selection, the session id, the transcript entry
// array (see §4.0 for the shapes), in-flight/status tracking, and the
// evidence panel. Every /chat and /tickers call goes through the hardened
// client in ./api — request/response shapes pass through untouched
// (plan §7.1/§7.2): this file never adds, renames, or reshapes a field.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppShell from "./components/AppShell";
import BootSequence from "./components/BootSequence";
import SideRail from "./components/SideRail";
import LandingPanel from "./components/LandingPanel";
import Transcript from "./components/Transcript";
import Composer from "./components/Composer";
import EvidencePanel from "./components/EvidencePanel";
import Icon from "./components/Icon";
import { ApiError, fetchTickers, sendMessage } from "./api";
import { predictMode } from "./lib/mode";
import { companyName } from "./lib/format";
import { useMediaQuery } from "./lib/useMediaQuery";

// The plan's >=1180px "wide" tier (§3.3) is also the point at which the
// evidence panel switches from a single drawer instance (AppShell's third
// grid column) to an inline instance rendered by whichever row owns the
// open evidence source. Keep this in sync with shell.css's own
// `@media (max-width: 1179px)` collapse — they encode the same boundary.
const WIDE_VIEWPORT_QUERY = "(min-width: 1180px)";

// Fixes defect 23: a /tickers failure must never brick the app. This list is
// only ever shown marked as cached — never presented as live data.
const FALLBACK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"];

function summarizeReply(reply) {
  if (reply && reply.type === "signal" && reply.signal) {
    const direction = String(reply.signal.signal ?? "unknown");
    const label = direction.charAt(0).toUpperCase() + direction.slice(1);
    const confidence =
      typeof reply.signal.confidence === "number" ? reply.signal.confidence.toFixed(2) : "—";
    return `${label} signal, confidence ${confidence}`;
  }
  if (reply && reply.type === "answer") {
    const n = Array.isArray(reply.citations) ? reply.citations.length : 0;
    return `Answer received, ${n} source${n === 1 ? "" : "s"}`;
  }
  return "Response received";
}

export default function App() {
  const [tickers, setTickers] = useState([]);
  const [tickersAreCached, setTickersAreCached] = useState(false);
  const [ticker, setTicker] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [entries, setEntries] = useState([]);
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState("connecting");
  const [evidence, setEvidence] = useState(null);
  const [announce, setAnnounce] = useState("");
  const [booted, setBooted] = useState(false);
  const isWideViewport = useMediaQuery(WIDE_VIEWPORT_QUERY);

  const lastFocusedRef = useRef(null);

  const loadTickers = useCallback((isRetry) => {
    setStatus((prev) => (isRetry ? "connecting" : prev));
    return fetchTickers()
      .then((list) => {
        setTickers(list);
        setTicker((prev) => (prev && list.includes(prev) ? prev : (list[0] ?? null)));
        setTickersAreCached(false);
        setStatus("online");
      })
      .catch(() => {
        // Only fall back to the hardcoded list on the very first load — a
        // failed retry after real tickers were already loaded should not
        // discard a coverage list that is still perfectly usable.
        setTickers((prev) => (prev.length > 0 ? prev : FALLBACK_TICKERS));
        setTicker((prev) => prev ?? FALLBACK_TICKERS[0]);
        setTickersAreCached((prevCached) => prevCached || true);
        setStatus("offline");
      });
  }, []);

  useEffect(() => {
    loadTickers(false);
    // Boot only — retries are user-triggered via onRetryConnect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retryConnect = useCallback(() => {
    loadTickers(true);
  }, [loadTickers]);

  const handleTickerChange = useCallback((next) => {
    setTicker(next);
    setSessionId(null);
    setEntries((prev) =>
      prev.length > 0
        ? [
            {
              id: crypto.randomUUID(),
              role: "notice",
              at: Date.now(),
              message: `session reset · now covering ${next}`,
            },
          ]
        : [],
    );
  }, []);

  const handleSend = useCallback(
    (rawText) => {
      const text = String(rawText ?? "").trim();
      if (!text || !ticker || pending) return;

      const predictedMode = predictMode(text);
      const userEntry = {
        id: crypto.randomUUID(),
        role: "user",
        at: Date.now(),
        text,
        predictedMode,
      };
      const pendingId = crypto.randomUUID();
      const pendingEntry = {
        id: pendingId,
        role: "pending",
        at: Date.now(),
        expectedMode: predictedMode,
        startedAt: Date.now(),
      };

      setEntries((prev) => [...prev, userEntry, pendingEntry]);
      setPending(true);

      sendMessage(sessionId, ticker, text)
        .then((reply) => {
          setSessionId(reply?.session_id ?? null);
          setStatus("online");
          setEntries((prev) =>
            prev.map((entry) =>
              entry.id === pendingId
                ? { id: pendingId, role: "assistant", at: Date.now(), payload: reply }
                : entry,
            ),
          );
          setAnnounce(summarizeReply(reply));
        })
        .catch((err) => {
          const isApiError = err instanceof ApiError;
          const reason = isApiError ? err.reason : "network";
          const message = isApiError ? err.message : String(err?.message ?? err ?? "Request failed");
          if (reason === "network") setStatus("offline");
          setEntries((prev) =>
            prev.map((entry) =>
              entry.id === pendingId
                ? { id: pendingId, role: "failure", at: Date.now(), reason, message, retryText: text }
                : entry,
            ),
          );
          setAnnounce("Request failed");
        })
        .finally(() => {
          setPending(false);
        });
    },
    [ticker, pending, sessionId],
  );

  const handleRetry = useCallback(
    (text) => {
      setEntries((prev) => prev.filter((e) => !(e.role === "failure" && e.retryText === text)));
      handleSend(text);
    },
    [handleSend],
  );

  const openEvidence = useCallback((source) => {
    lastFocusedRef.current =
      typeof document !== "undefined" ? document.activeElement : null;
    setEvidence(source);
  }, []);

  const closeEvidence = useCallback(() => {
    setEvidence(null);
    const el = lastFocusedRef.current;
    if (el && typeof el.focus === "function") el.focus();
    lastFocusedRef.current = null;
  }, []);

  useEffect(() => {
    if (!evidence) return undefined;
    function onKeyDown(e) {
      if (e.key === "Escape") closeEvidence();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [evidence, closeEvidence]);

  const stats = useMemo(() => {
    let lookups = 0;
    let signals = 0;
    for (const entry of entries) {
      if (entry.role !== "assistant") continue;
      if (entry.payload?.type === "answer") lookups += 1;
      else if (entry.payload?.type === "signal") signals += 1;
    }
    return { entries: entries.length, lookups, signals };
  }, [entries]);

  return (
    <>
      {!booted && <BootSequence onDone={() => setBooted(true)} />}
      <a className="skip-link" href="#transcript">
        Skip to transcript
      </a>
      <AppShell
        evidenceOpen={isWideViewport && evidence != null}
        rail={
          <SideRail
            tickers={tickers}
            activeTicker={ticker}
            onSelectTicker={handleTickerChange}
            status={status}
            sessionId={sessionId}
            stats={stats}
            onRetryConnect={retryConnect}
            tickersAreCached={tickersAreCached}
          />
        }
        contextBar={
          <div className="shell__contextbar t-meta">
            <span className="shell__contextbar-ticker">
              {ticker ? `${ticker} · ${companyName(ticker)}` : "No ticker selected"}
            </span>
            <span className="shell__contextbar-legend">
              <span className="shell__legend-item shell__legend-item--lookup">
                <Icon name="arrow-right" size={12} />
                GROUNDED LOOKUP
              </span>
              <span className="shell__legend-item shell__legend-item--signal">
                <Icon name="square" size={8} />
                DIRECTIONAL CALL
              </span>
            </span>
          </div>
        }
        composer={
          <Composer onSend={handleSend} disabled={!ticker || pending} loading={pending} ticker={ticker} />
        }
        evidencePanel={
          // >=1180px: a single drawer instance lives in AppShell's third grid
          // column. Below that, EvidencePanel is rendered inline by whichever
          // TranscriptEntry row owns the open source instead (see Transcript
          // -> TranscriptEntry -> AnswerBody/SignalCard) — nothing is
          // rendered here, so the shell never reserves the third column and
          // never adds a stray grid item once shell.css collapses to two
          // columns at <1180px.
          isWideViewport && evidence ? (
            <EvidencePanel source={evidence} onClose={closeEvidence} variant="drawer" />
          ) : null
        }
      >
        {entries.length === 0 ? (
          <main id="transcript" className="shell__landing-slot">
            <LandingPanel
              ticker={ticker}
              tickers={tickers}
              onUsePrompt={handleSend}
              offline={status === "offline"}
              onRetryConnect={retryConnect}
              tickersAreCached={tickersAreCached}
            />
          </main>
        ) : (
          <Transcript
            entries={entries}
            ticker={ticker}
            onRetry={handleRetry}
            onOpenEvidence={openEvidence}
            evidence={evidence}
            isWideViewport={isWideViewport}
            onCloseEvidence={closeEvidence}
          />
        )}
      </AppShell>
      <p className="visually-hidden" aria-live="polite">
        {announce}
      </p>
    </>
  );
}
