// HTTP client for the FastAPI backend (api/main.py). Contract is frozen —
// see frontend-design-plan.md §4.8 and §7.1/§7.2: request/response shapes
// pass through untouched, no field renaming or normalization.
//
// Hardening added here (defect 21): AbortController-based timeouts and a
// typed ApiError, so a hung or unreachable backend never hangs the UI
// forever. Both functions keep their original call signatures — an optional
// trailing AbortSignal is the only addition — so existing callers work
// unchanged.

// Default is byte-identical to the previous hardcoded value, so local dev
// and the backend's CORS allow-list (http://localhost:5173) are unaffected
// when VITE_API_BASE isn't set.
const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const CHAT_TIMEOUT_MS = 45000; // signal generation runs a full hybrid sweep — genuinely slow
const TICKERS_TIMEOUT_MS = 8000;

export class ApiError extends Error {
  constructor(reason, message, status = null, statusText = null) {
    super(message);
    this.name = "ApiError";
    this.reason = reason; // "network" | "timeout" | "server"
    this.status = status;
    this.statusText = statusText;
  }
}

async function requestJson(url, { method = "GET", body, timeoutMs, signal } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let onExternalAbort;
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      onExternalAbort = () => controller.abort();
      signal.addEventListener("abort", onExternalAbort);
    }
  }

  try {
    let res;
    try {
      res = await fetch(url, {
        method,
        headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err) {
      if (err?.name === "AbortError") {
        throw new ApiError("timeout", `${method} ${url} timed out after ${timeoutMs}ms`);
      }
      throw new ApiError("network", err?.message || "network request failed");
    }

    if (!res.ok) {
      throw new ApiError("server", `${method} ${url} returned ${res.status}`, res.status, res.statusText);
    }

    try {
      return await res.json();
    } catch (err) {
      throw new ApiError("network", err?.message || "invalid response body");
    }
  } finally {
    clearTimeout(timeoutId);
    if (signal && onExternalAbort) signal.removeEventListener("abort", onExternalAbort);
  }
}

// Response body (the raw /chat reply) passes through completely untouched.
export async function sendMessage(sessionId, ticker, message, signal) {
  return requestJson(`${BASE}/chat`, {
    method: "POST",
    body: { session_id: sessionId, ticker, message },
    timeoutMs: CHAT_TIMEOUT_MS,
    signal,
  });
}

export async function fetchTickers(signal) {
  const data = await requestJson(`${BASE}/tickers`, {
    timeoutMs: TICKERS_TIMEOUT_MS,
    signal,
  });
  return data.tickers;
}
