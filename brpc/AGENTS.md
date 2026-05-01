# AGENTS.md

`brpc/` is the paper-facing public namespace.

- Keep this package lightweight.
- Re-export stable user-facing APIs from `calib` rather than moving algorithm internals here.
- Add small registries or convenience wrappers here only when they make clone-and-run usage clearer.
- Do not put experiment implementation logic in this folder; use the explicit runners under `calib` and `tools`.
