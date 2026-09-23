# Earnings Whisper — Frontend

A research-terminal chat UI over a retrieval-augmented (RAG) earnings-signal
API. Ask a grounded question about a company's latest filing and get a cited
answer, or request a directional call (bullish / bearish / flat) backed by
supporting passages pulled from the filing.

## Prerequisites

- Node.js
- The backend API, running locally (see below)

## Running the backend

This frontend talks to a backend API expected at `http://localhost:8000`.
From the **repo root** (not `frontend/`):

```
uvicorn api.main:app --reload
```

The backend's CORS policy allows only `http://localhost:5173`, which is
Vite's default dev port — keep the frontend on that port for local dev.

## Running the frontend

```
npm install
npm run dev
```

This starts the Vite dev server at `http://localhost:5173`.

If the backend isn't running, the app still loads: it falls back to a
cached ticker list and shows the API as offline, with a retry control.

## Configuration

The API base URL is controlled by the `VITE_API_BASE` environment
variable — see `.env.example`. It defaults to `http://localhost:8000` when
unset, so no `.env` file is required for the default local setup.

Google sign-in and per-ticker chat-history persistence are optional and
additive — with no `VITE_FIREBASE_*` vars set (see `.env.example`), the app
works exactly as it does signed out. Details in
`../docs/superpowers/specs/2026-09-22-firebase-chat-persistence-design.md`.

## Linting

```
npm run lint
```

Runs oxlint over the project.

## Deployment

Pushes to `main` that touch this directory build and deploy it to GitHub
Pages via `../.github/workflows/deploy-pages.yml`. `vite.config.js` sets
`base: '/EarningsWhisper/'` only for `npm run build` (not `npm run dev`) to
match that URL, and `.env.production` bakes in the live Firebase config
that build uses — see the root README's "Deployment" section.
