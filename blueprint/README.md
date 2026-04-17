# Blueprint

Interactive web visualization of the `LeanMlir/Proofs/` suite. Each node
in the dependency graph links back to its Lean declaration; click around
to navigate the proof compositionally from primitives to the ViT finale.

## CI build (automatic)

Pushing to `main` triggers `.github/workflows/blueprint.yml`:

1. Builds the Lean project.
2. Runs `leanblueprint checkdecls` — verifies every `\lean{…}` reference
   points at a real Lean declaration.
3. Runs `leanblueprint web` — compiles `src/content.tex` into HTML with
   an interactive dependency DAG.
4. Runs `doc-gen4` — generates Lean API docs.
5. Deploys everything to GitHub Pages: `home_page/` → `/`, blueprint → `/blueprint/`, docs → `/docs/`.

Enable GitHub Pages in the repo settings (Settings → Pages → Source: GitHub Actions) for the first deploy to work.

## Local build

Requires a full TeX Live installation plus graphviz.

```bash
sudo apt-get install texlive-full graphviz
pip install leanblueprint

leanblueprint checkdecls       # verify \lean{…} refs
leanblueprint pdf              # build PDF → blueprint/print/*.pdf
leanblueprint web              # build HTML → blueprint/web/index.html
leanblueprint serve            # local web server on :8000
```

## Structure

```
blueprint/
├── src/
│   ├── content.tex            ← every axiom/theorem, with \lean{…}+\uses{…}
│   ├── macros/common.tex      ← shared math notation
│   ├── web.tex / print.tex    ← entry points for HTML / PDF builds
│   └── plastex.cfg            ← plasTeX config
├── lean_decls                 ← auto-generated list of cited Lean decls
└── README.md                  ← this file
```

## Adding a new theorem to the blueprint

```latex
\begin{theorem}[Short description]
  \label{thm:my_new_theorem}
  \lean{Proofs.my_new_theorem}
  \leanok
  \uses{thm:dependency_1, ax:primitive_axiom}
  Optional math content; the key fields are \lean, \leanok, and \uses.
\end{theorem}
```

- `\lean{…}` — the exact Lean declaration name (tied to the source).
- `\leanok` — mark as complete. Removed if the theorem is still in progress.
- `\uses{…}` — other labels this theorem depends on; drives the DAG edges.
