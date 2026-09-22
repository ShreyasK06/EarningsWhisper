// Firestore-backed chat history, scoped per (user, ticker). See
// docs/superpowers/specs/2026-09-22-firebase-chat-persistence-design.md.
//
// entryToDocData/docDataToEntry are pure (no Firestore dependency) so the
// entry<->document translation is unit-testable without a live project —
// only loadChatHistory/saveMessage/PERSISTED_ROLES touch the SDK.
import {
  collection,
  doc,
  getDocs,
  orderBy,
  query,
  setDoc,
  Timestamp,
} from "firebase/firestore";
import { db, firebaseConfigured } from "./firebase";

// pending/failure are transient UI state, never persisted — see the design
// spec's "Never persisted" note.
export const PERSISTED_ROLES = new Set(["user", "assistant", "notice"]);

function messagesPath(uid, ticker) {
  return `users/${uid}/chats/${ticker}/messages`;
}

// entry.at is a client Date.now() millisecond number (see App.jsx) — stored
// as a real Firestore Timestamp so `orderBy("at")` sorts correctly, not as
// a raw number (which would sort as a string if ever queried loosely).
// `_id` is dropped rather than stored — it becomes the Firestore doc ID
// itself (see saveMessage), not a field within the document.
export function entryToDocData({ id: _id, at, ...rest }) {
  return { ...rest, at: Timestamp.fromMillis(at) };
}

export function docDataToEntry(id, data) {
  const { at, ...rest } = data;
  return { id, ...rest, at: at instanceof Timestamp ? at.toMillis() : at };
}

export async function loadChatHistory(uid, ticker) {
  if (!firebaseConfigured || !uid || !ticker) return [];
  try {
    const q = query(collection(db, messagesPath(uid, ticker)), orderBy("at"));
    const snapshot = await getDocs(q);
    return snapshot.docs.map((d) => docDataToEntry(d.id, d.data()));
  } catch (err) {
    // A load failure must never block the app — fall back to an empty
    // transcript (same as logged-out) rather than an error screen.
    console.error("Failed to load chat history:", err);
    return [];
  }
}

export async function saveMessage(uid, ticker, entry) {
  if (!firebaseConfigured || !uid || !ticker) return;
  if (!PERSISTED_ROLES.has(entry.role)) return;
  try {
    await setDoc(doc(db, messagesPath(uid, ticker), entry.id), entryToDocData(entry));
  } catch (err) {
    // The message is already visible in the UI (appended to local state
    // before this call) — a save failure just means it won't survive a
    // reload, never a broken chat experience right now.
    console.error("Failed to save chat message:", err);
  }
}
