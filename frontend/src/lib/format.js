// Formatting helpers shared across the transcript, rail, landing panel, and
// signal card. Deliberately dependency-free — see frontend-design-plan.md §5
// (dayjs/date-fns rejected; Intl.DateTimeFormat covers both formats needed).

// Real company names for the known coverage set. A ticker absent from this
// map (e.g. a 6th ticker returned by /tickers) must never throw — callers
// fall back to the bare symbol via `companyName()` or `COMPANY_NAMES[t] ?? t`.
export const COMPANY_NAMES = {
  AAPL: "Apple Inc.",
  MSFT: "Microsoft Corp.",
  NVDA: "NVIDIA Corp.",
  AMZN: "Amazon.com Inc.",
  GOOGL: "Alphabet Inc.",
};

/** Returns the known company name for `ticker`, or the ticker itself. */
export function companyName(ticker) {
  return COMPANY_NAMES[ticker] ?? ticker;
}

// formatToParts + manual assembly, rather than a full formatted string, so
// output is stable regardless of the runtime's default locale (Node/browser
// default-locale formatting for hour12/month order is not guaranteed).
const TIME_PARTS = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const DATE_PARTS = new Intl.DateTimeFormat("en-US", {
  day: "2-digit",
  month: "short",
  year: "numeric",
});

function toDate(ts) {
  const date = ts == null ? new Date() : new Date(ts);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** `ts` (epoch ms, ISO string, or Date) -> "14:22". Never throws. */
export function clockTime(ts) {
  const date = toDate(ts);
  if (!date) return "--:--";
  const parts = TIME_PARTS.formatToParts(date);
  const hour = parts.find((p) => p.type === "hour")?.value ?? "00";
  const minute = parts.find((p) => p.type === "minute")?.value ?? "00";
  return `${hour}:${minute}`;
}

/** `ts` (epoch ms, ISO string, or Date) -> "15 SEP 2026". Never throws. */
export function dateline(ts) {
  const date = toDate(ts);
  if (!date) return "--- --- ----";
  const parts = DATE_PARTS.formatToParts(date);
  const day = parts.find((p) => p.type === "day")?.value ?? "";
  const month = parts.find((p) => p.type === "month")?.value ?? "";
  const year = parts.find((p) => p.type === "year")?.value ?? "";
  return `${day} ${month} ${year}`.toUpperCase();
}

/** Zero-pads a number to 2 digits: pad2(4) -> "04". */
export function pad2(n) {
  return String(n).padStart(2, "0");
}

/** Clamps a number into [0, 1]; non-numeric input clamps to 0. */
export function clamp01(n) {
  const num = Number(n);
  if (Number.isNaN(num)) return 0;
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}

/** Truncates `str` to at most `n` characters, appending an ellipsis. */
export function truncate(str, n) {
  const s = String(str ?? "");
  if (n <= 0) return "";
  if (s.length <= n) return s;
  return `${s.slice(0, Math.max(0, n - 1))}…`;
}
