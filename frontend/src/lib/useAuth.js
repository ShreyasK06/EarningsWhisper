import { useCallback, useEffect, useState } from "react";
import { signInWithPopup, signOut as firebaseSignOut, onAuthStateChanged } from "firebase/auth";
import { auth, googleProvider, firebaseConfigured } from "./firebase";

// Current Firebase Auth user + Google sign-in/out actions. Returns
// user: null when signed out OR when Firebase isn't configured yet (see
// firebase.js) — callers don't need to check firebaseConfigured themselves,
// signed-out behavior is the correct fallback either way.
export function useAuth() {
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(!firebaseConfigured);

  useEffect(() => {
    if (!firebaseConfigured) return undefined;
    return onAuthStateChanged(auth, (nextUser) => {
      setUser(nextUser);
      setAuthReady(true);
    });
  }, []);

  const signInWithGoogle = useCallback(() => {
    if (!firebaseConfigured) return Promise.resolve();
    return signInWithPopup(auth, googleProvider).catch((err) => {
      // A closed popup or a network hiccup is routine, not exceptional —
      // log it and let the UI stay in its current (signed-out) state
      // rather than surfacing a blocking error for what's usually just
      // the user changing their mind mid-flow.
      console.error("Google sign-in failed:", err);
    });
  }, []);

  const signOut = useCallback(() => {
    if (!firebaseConfigured) return Promise.resolve();
    return firebaseSignOut(auth).catch((err) => {
      console.error("Sign-out failed:", err);
    });
  }, []);

  return { user, authReady, signInWithGoogle, signOut };
}
