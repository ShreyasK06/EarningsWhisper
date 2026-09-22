// Firebase app init — Auth (Google sign-in) + Firestore (chat persistence).
// See docs/superpowers/specs/2026-09-22-firebase-chat-persistence-design.md.
//
// All VITE_FIREBASE_* values are Firebase's public web app config, not
// secrets — real access control lives entirely in Firestore security rules
// (firestore.rules), not in hiding this config. Safe to ship in the bundle.
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// True once real config values are present. Guards every Firebase call site
// (useAuth, chatHistory) so a checkout with no Firebase project configured
// yet degrades to "chat persistence is unavailable" instead of throwing at
// import time — matching this project's existing "a missing backend piece
// never bricks the app" philosophy (see api.js's ApiError handling).
export const firebaseConfigured = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId);

export const app = firebaseConfigured ? initializeApp(firebaseConfig) : null;
export const auth = firebaseConfigured ? getAuth(app) : null;
export const db = firebaseConfigured ? getFirestore(app) : null;
export const googleProvider = new GoogleAuthProvider();
