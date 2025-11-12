# Repository Guidelines

## Project Structure & Module Organization
The workspace is intentionally small: `download.py` is the main Sentinel Hub ingestion script, `space.ipynb` holds exploratory analysis, and `downloads/` is the cache for fetched tiles and derived TIFFs/PNGs. Keep intermediate artifacts inside `downloads/` (or subfolders under it) so they remain git-ignored, and stage only reproducible code or notebooks. When extending functionality, prefer new modules in the repo root (for example `processing/ndvi.py`) so imports stay relative and lightweight.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — isolate dependencies before installing remote-sensing libraries.
- `pip install sentinelhub matplotlib numpy` — minimal toolchain needed by `download.py`; pin exact versions when experiments stabilize.
- `python download.py` — runs the configured AOI pull, writes imagery into `downloads/`, and opens a matplotlib preview.
- `jupyter lab space.ipynb` — iterate on downstream experiments; keep heavy notebooks out of commits by clearing outputs.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions, and UpperCamelCase only for classes. Declare configuration constants (bbox, AOI metadata, environment variables) near the top of each module and document units in inline comments. Use explicit typing (`def fetch_scene(...) -> np.ndarray`) for any helper you add. Keep plotting logic separated from data fetching so modules stay testable.

## Testing Guidelines
There is no formal suite yet; add `tests/` with `pytest` as you introduce reusable helpers. Name files `test_<module>.py` and mirror the structure inside `download.py`. For scripts that hit remote APIs, stub Sentinel Hub calls via fixtures and guard network-dependent tests behind `pytest -m integration`. Before opening a PR, execute `python download.py` with a small bbox to confirm credentials, output resolution, and directory permissions still work.

## Commit & Pull Request Guidelines
Recent history (`add syllabus...`, `initial commit ...`) shows short, imperative subjects; keep following that pattern (“add tiling helper”, “refine bbox parser”). Reference issues in the body when applicable, describe AOI/resolution changes, and paste sample command output or thumbnails if behavior changes. PRs should include: scope summary, verification steps (commands run), any Sentinel Hub quota impact, and mention of new secrets or config files so reviewers can update their `.env`.

## Security & Configuration Tips
Never hard-code Sentinel Hub credentials; rely on `SH_CLIENT_ID`, `SH_CLIENT_SECRET`, and `SH_INSTANCE_ID` exported in your shell or stored in `.env` that stays untracked. Before pushing notebooks, ensure they do not embed tokens or exact coordinates that should remain private. Rotate credentials immediately if they accidentally appear in git history.
