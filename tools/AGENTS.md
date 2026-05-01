# AGENTS.md

`tools/` contains orchestration, plotting, aggregation, and post-processing helpers.

- Keep helpers runnable from the repository root.
- Prefer explicit CLI flags over hidden constants.
- If a helper becomes a paper experiment entrypoint, document the exact command in `../rolled_cusum_modeling_workdoc.md` and `../README.md`.
- Do not import from notebook-only code.
