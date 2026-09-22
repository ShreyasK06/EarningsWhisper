// Eight hard-coded inline glyphs on a 16x16 grid, one consistent stroke
// style. Paths are inlined in JSX rather than loaded from a sprite file:
// a <use href="/icons.svg#x"> reference breaks under a non-root deploy base,
// and this project's conventions forbid hardcoded absolute asset paths in
// Vite projects. See frontend-design-plan.md §4.7 and §5.
//
// Callers put the accessible name on the *wrapping* control (a button's
// aria-label, etc.) — the icon itself stays aria-hidden by default.

const PATHS = {
  "arrow-right": <path d="M2.5 8h10.5M9 4l4 4-4 4" />,
  "arrow-up-right": <path d="M4.5 11.5l7-7M6 4.5h5.5v5.5" />,
  "chevron-down": <path d="M4 6l4 4 4-4" />,
  check: <path d="M3.5 8.5l3 3 6-7" />,
  close: (
    <>
      <path d="M4 4l8 8" />
      <path d="M12 4l-8 8" />
    </>
  ),
  retry: (
    <>
      <path d="M13 7A5 5 0 1 1 11.9 3.9" />
      <path d="M13 3.5V7h-3.5" />
    </>
  ),
  square: <rect x="4.5" y="4.5" width="7" height="7" />,
  quote: (
    <>
      <path d="M5.6 9c1.3-.5 1.8-1.6 1.8-2.9 0-.85-.55-1.5-1.35-1.5S4.7 5.25 4.7 6.1c0 .65.4 1.1.95 1.3-.2.65-.65 1.2-1.4 1.5" />
      <path d="M10.7 9c1.3-.5 1.8-1.6 1.8-2.9 0-.85-.55-1.5-1.35-1.5S9.8 5.25 9.8 6.1c0 .65.4 1.1.95 1.3-.2.65-.65 1.2-1.4 1.5" />
    </>
  ),
  sun: (
    <>
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.5M8 13v1.5M14.5 8H13M3 8H1.5M12.36 3.64l-1.06 1.06M4.7 11.3l-1.06 1.06M12.36 12.36l-1.06-1.06M4.7 4.7 3.64 3.64" />
    </>
  ),
  moon: <path d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z" />,
  palette: (
    <>
      <path d="M8 1.5a6.5 6.5 0 1 0 0 13c.83 0 1.5-.67 1.5-1.5 0-.4-.16-.76-.4-1.03a1.5 1.5 0 0 1 1.1-2.47H11.5A3 3 0 0 0 14.5 6.5C14.5 3.7 11.6 1.5 8 1.5z" />
      <circle cx="5" cy="7" r=".9" fill="currentColor" stroke="none" />
      <circle cx="6.5" cy="10.5" r=".9" fill="currentColor" stroke="none" />
      <circle cx="10" cy="5.5" r=".9" fill="currentColor" stroke="none" />
    </>
  ),
};

export default function Icon({
  name,
  size = 16,
  className,
  "aria-hidden": ariaHidden = "true",
  focusable = "false",
}) {
  const path = PATHS[name];
  if (!path) return null;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden={ariaHidden}
      focusable={focusable}
    >
      {path}
    </svg>
  );
}
