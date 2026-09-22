// Replaces the old global error banner (fixes defect 22 — a failed /chat
// call used to orphan the user's message with no inline recovery). See
// frontend-design-plan.md §4.2. Copy is direct and active by design — no
// "Oops", no exclamation marks.

import Icon from "./Icon";

const REASON_COPY = {
  network: {
    heading: "Couldn't reach the API.",
    body: "The request to localhost:8000 failed. The backend may not be running.",
  },
  timeout: {
    heading: "No response after 45 seconds.",
    body: "Signal generation can be slow, but this exceeded the timeout.",
  },
  server: {
    heading: "The API returned an error.",
    body: null, // uses the caller-supplied `message` (the status text) verbatim
  },
};

export default function FailureEntry({ reason, message, retryText, onRetry }) {
  const copy = REASON_COPY[reason];
  const heading = copy?.heading ?? "The request failed.";
  const body = copy?.body ?? message ?? "An unexpected error occurred.";

  return (
    <div className="failure">
      <p className="failure__heading t-ui">{heading}</p>
      <p className="failure__body t-body-sm">{body}</p>
      <p className="failure__command-label t-micro">START THE BACKEND</p>
      <pre className="failure__command t-micro">uvicorn api.main:app --reload</pre>
      <button type="button" className="failure__retry t-ui" onClick={() => onRetry?.(retryText)}>
        <Icon name="retry" size={14} />
        <span>Retry</span>
      </button>
    </div>
  );
}
