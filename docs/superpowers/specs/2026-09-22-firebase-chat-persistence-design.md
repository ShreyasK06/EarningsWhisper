# Firebase chat persistence + Google sign-in — Design

**Date:** 2026-09-22
**Status:** Approved by user in chat (sectioned design walkthrough), implementation follows this spec.

## Goal

Let a logged-in user's chat history persist across sessions/devices, scoped per ticker. Logged-out use is unaffected — the app must keep working exactly as it does today (ephemeral, in-memory) when no one is signed in.

## Decisions made (via brainstorming Q&A)

1. **Provider: Firebase** (Auth + Firestore), not Supabase.
2. **Sign-in method: Google only**, via Firebase Auth's Google provider. No email/password.
3. **Who writes to Firestore: the frontend, directly**, via the Firebase JS SDK. The FastAPI backend (`api/main.py`) is untouched — it has no Firebase awareness, no service-account credentials, no token verification. This is the smaller, more idiomatic-for-Firebase option; the alternative (backend writes via `firebase-admin`, frontend sends an ID token per request) was explicitly declined.
4. **History scope: one running history per ticker**, not named/listed sessions. Switching to a ticker (when signed in) loads that ticker's saved messages; new messages append to it. This mirrors today's per-ticker reset behavior, except persisted instead of wiped.

## Architecture

```
Frontend (React)
  |- Firebase Auth (Google provider) - sign in/out, current user
  \- Firestore - read/write chat messages directly from the browser,
     gated by security rules (server-side enforcement, not just
     client-side trust)

FastAPI backend - UNCHANGED. No Firebase code, no new endpoints.
```

**Why frontend-direct instead of backend-mediated:** the backend's job (retrieval + generation) and the persistence job (who-said-what, scoped to a signed-in user) are orthogonal. Routing persistence through the backend would mean managing a Firebase service account there, verifying ID tokens on every `/chat` call, and duplicating the Firestore schema knowledge in two codebases for no functional gain — the frontend already knows the full entry shape it wants to save, and Firestore security rules give the same access-control guarantee (a user can only touch their own data) without a backend round-trip.

## Data model

```
users/{uid}/chats/{ticker}/messages/{messageId}
  role: "user" | "assistant" | "notice"
  at: Firestore Timestamp
  # remaining fields mirror today's in-memory entry shape (App.jsx):
  #   role "user":      text, predictedMode
  #   role "assistant":  payload (the raw /chat reply: {type, ...})
  #   role "notice":     message
```

One subcollection per `(user, ticker)` pair, documents ordered by `at`. Reusing the existing in-memory entry shape (not inventing a new one) means a loaded Firestore document can be dropped straight into the `entries` array the UI already renders — no translation layer.

**Never persisted:** `pending` and `failure` role entries. Those are transient UI states (a request in flight, a request that failed) — persisting them would mean a signed-in user reloading the app sees a stuck spinner or a stale error instead of just the real conversation. Only a `user` message and its resolved `assistant` reply get written, and only once the reply actually arrives.

## Auth UI

A control in `SideRail`'s wordmark row, alongside the appearance controls added earlier this session:
- **Signed out:** a "Sign in with Google" button (opens Firebase's Google popup/redirect flow).
- **Signed in:** the user's Google avatar (or initial, if no photo) + display name, with a sign-out affordance (click to reveal, matching the accent-picker's disclosure pattern already in place).

## Data flow

- **On mount / ticker change**, if a user is signed in: query `users/{uid}/chats/{ticker}/messages` ordered by `at`, hydrate `entries` from the results instead of starting empty. If signed out, behave exactly as today (empty transcript on ticker switch).
- **On a successful exchange** (the existing `handleSend` success path in `App.jsx`), if signed in: write the user message and the assistant reply to Firestore as two documents (using the same `crypto.randomUUID()` already generated for each entry as the document ID, so an entry's Firestore doc ID and its in-memory React key are the same value).
- **On sign-in**, if the currently active ticker has saved history, load it immediately (don't wait for the user to switch tickers).
- **On sign-out**, clear `entries` back to empty (the same landing state as picking a fresh ticker) rather than leaving another user's data on screen.

## Error handling

A Firestore write or read failure must never break the chat UI — this matches the project's existing philosophy (`api.js`'s `ApiError` handling, `App.jsx`'s tickers-fetch fallback). If a save fails: the message the user already sees in the transcript stays exactly as it is (it was appended to local state before the write was attempted); log the failure to the console; do not retry automatically, do not surface a blocking error to the user. If a history load fails: fall back to an empty transcript (same as logged-out), not an error screen.

## What requires manual setup (cannot be done by the assistant)

1. Create a Firebase project in the Firebase Console.
2. Enable **Google** as a sign-in provider under Firebase Authentication.
3. Create a **Firestore** database (production mode, not test mode — the security rules below are what actually gate access).
4. Copy the web app's Firebase config (`apiKey`, `authDomain`, `projectId`, `storageBucket`, `messagingSenderId`, `appId`) into `frontend/.env` under the `VITE_FIREBASE_*` keys this implementation adds to `.env.example`. Firebase's client config is not a secret (it's safe to ship in a browser bundle) — real access control comes entirely from the security rules, not from hiding this config.
5. Publish the Firestore security rules this implementation writes to a `firestore.rules` file, either by pasting them into the Firebase Console's Rules editor or via `firebase deploy --only firestore:rules` if the user has the Firebase CLI authenticated locally.

## Security rules (Firestore)

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId}/chats/{ticker}/messages/{messageId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

## Testing

- Frontend: unit-test the pure translation logic (entry <-> Firestore document shape) with a mocked Firestore client, following the existing `tests/generation/test_llm_client.py`-style fake-object pattern already used on the Python side — no live Firebase project needed for this.
- Manual/live verification (once the user has completed the manual setup above): sign in, send a message, reload the page, confirm the message is still there; sign out, confirm the transcript clears and no Firestore calls happen; switch tickers while signed in, confirm each ticker's history is independent.

## Out of scope (YAGNI, not requested)

- Editing or deleting individual saved messages.
- Sharing a chat between users.
- Any auth method beyond Google.
- Named/listed chat sessions (explicitly declined in favor of one running history per ticker).
- Exporting chat history.
