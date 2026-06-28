# QFlex Distribution — Interactive Tutorial

A live, in-browser interactive companion to *The QFlex Distribution* (Bickel et al.),
built as a [marimo](https://marimo.io) notebook and a tutorial for the
[`qflex`](https://pypi.org/project/qflex/) Python library.

**▶ Live notebook:** https://USERNAME.github.io/REPO/  *(update after enabling Pages)*

The page runs entirely in your browser via WebAssembly (Pyodide) — no server, no
install. Move the sliders and dropdowns and the figures recompute live.

## What's inside

| Part | Paper § | Topic |
|------|---------|-------|
| 1 | §1 | Motivation and a running example |
| A | §2 | Constructing valid quantile functions (Gilchrist rules) |
| B | §3 | The QFlex system: bases, fitting, interpolation, moments |
| C | §4 | Monotonicity: Theorems 1 & 5, Propositions 2–4 |
| D | §5 | Modality and the quantile curvature function |
| E | §6 | Bounded and semi-bounded variants |
| F | §7 | QFlex vs Metalog across distributions + skew/kurtosis explorer |
| G | §8 | Demonstrating monotonicity control |
| H | §9 | Conclusion and references |

## Deploy it (GitHub Pages)

1. Create a new GitHub repo and push these files (see commands below).
2. In the repo: **Settings → Pages → Source → "GitHub Actions."**
3. Push to `main`. The included workflow exports the notebook to WebAssembly and
   publishes it. The live URL appears in the Actions run (and under Settings → Pages).

```bash
git init
git add .
git commit -m "QFlex interactive marimo tutorial"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

## Run / edit locally

```bash
pip install -r requirements.txt
marimo edit qflex_tutorial.py      # interactive editor
# or preview the exact WASM build that gets deployed:
marimo export html-wasm qflex_tutorial.py -o _site --mode run
python -m http.server --directory _site   # then open the printed URL in a browser
```

## How the browser build works

`qflex` and the vendored Metalog (`_metalog`) are local packages, so they are
shipped as pre-built wheels in [`public/`](public/). The notebook's first cell
installs them with `micropip` when it detects a Pyodide runtime; locally that
cell is a no-op. `cvxpy` is intentionally **not** required in the browser — when
it is absent, `qflex` falls back to an equivalent `scipy.optimize` formulation of
the tail–center (Proposition 4) solver, which produces identical results.

### Refreshing the wheels

The wheels rarely change. To rebuild them after updating `qflex` or the vendored
Metalog, from a checkout of the `qflex` library repo:

```bash
pip install build

# qflex wheel (built straight from the library's pyproject):
python -m build --wheel --outdir /path/to/qflex-tutorial/public

# metalog wheel: copy notebooks/_metalog into an empty dir with this pyproject.toml,
# then `python -m build --wheel --outdir /path/to/qflex-tutorial/public`:
#   [build-system]
#   requires = ["hatchling"]
#   build-backend = "hatchling.build"
#   [project]
#   name = "qflex_metalog_bundle"
#   version = "0.1.0"
#   requires-python = ">=3.9"
#   dependencies = ["numpy>=1.24", "scipy>=1.10", "matplotlib>=3.7"]
#   [tool.hatch.build.targets.wheel]
#   packages = ["_metalog"]
```

If the wheel **filenames** change (e.g. a new version), update the URLs in the
notebook's first cell (the WASM bootstrap) to match.
