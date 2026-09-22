// Pure layout shell — the 3-column grid from frontend-design-plan.md §3.2.
// Owns no state: App.jsx decides what goes in each slot and whether the
// evidence column is open. Keeping this component dumb is what lets it be
// implemented in parallel with the five sibling components it lays out.
//
// Slots:
//   rail          - <SideRail> (renders its own <aside>)
//   contextBar    - small header row (ticker · mode legend) above the transcript
//   children      - the transcript column body: either the landing state or
//                   <Transcript>, chosen by App.jsx based on entries.length
//   composer      - <Composer>
//   evidencePanel - <EvidencePanel> (renders its own <aside>, or nothing)
//   evidenceOpen  - whether the evidence column should reserve grid width;
//                   the column-width change itself is instant/untransitioned
//                   per §3.2 — only the panel's own content animates in.
export default function AppShell({
  rail,
  contextBar,
  composer,
  evidencePanel,
  evidenceOpen,
  children,
}) {
  return (
    <div className="shell" data-evidence={evidenceOpen ? "open" : undefined}>
      {rail}
      <div className="shell__main">
        {contextBar}
        {children}
        {composer}
      </div>
      {evidencePanel}
    </div>
  );
}
