import { useCallback, useEffect, useState } from "react";

const THEME_KEY = "earnings-whisper:theme";
const ACCENT_KEY = "earnings-whisper:accent";

// `swatch` is the light-mode hex for each preset (tokens.css :root / [data-accent]
// values) — used only to paint the picker's preview dots, which need a fixed
// color to show per-option regardless of which single accent is active
// document-wide; the dark-mode re-tuned hex is close enough in hue that a
// separate swatch set isn't worth the duplication.
export const ACCENTS = [
  { id: "navy", label: "Navy", swatch: "#1d3557" },
  { id: "forest", label: "Forest", swatch: "#2f5233" },
  { id: "plum", label: "Plum", swatch: "#5b3358" },
  { id: "rust", label: "Rust", swatch: "#7a3b26" },
  { id: "slate", label: "Slate", swatch: "#3d4a5c" },
];

function readStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // localStorage can throw in private-browsing/disabled-storage contexts —
    // theme just won't persist across reloads, never a hard failure.
  }
}

function prefersDark() {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

// Owns the two independent appearance axes (light/dark, accent color),
// applied to <html> as data-theme/data-accent so tokens.css can key off
// them with plain attribute selectors. Persists both to localStorage;
// falls back to the OS preference for the initial theme when nothing was
// stored yet, and to the default "navy" accent (unset attribute, since
// navy's values already live in :root with no override needed).
export function useTheme() {
  const [theme, setTheme] = useState(() => readStorage(THEME_KEY) ?? (prefersDark() ? "dark" : "light"));
  const [accent, setAccent] = useState(() => readStorage(ACCENT_KEY) ?? "navy");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    writeStorage(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (accent === "navy") {
      document.documentElement.removeAttribute("data-accent");
    } else {
      document.documentElement.setAttribute("data-accent", accent);
    }
    writeStorage(ACCENT_KEY, accent);
  }, [accent]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggleTheme, accent, setAccent };
}
