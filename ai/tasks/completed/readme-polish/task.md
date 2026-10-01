# Task: Review and polish README

## Goal

Perform a targeted review of `README.md` as a public portfolio and developer-facing project page. Improve navigation and the execution/configuration flow, clarify the implemented MIA, and update path-related statements after the switch to repository-local defaults.

## Sources of truth

- `docs/thesis-facts.md` for research design, numerical results, and scientific conclusions
- current source code, configuration, and `pyproject.toml` for implementation behavior
- `docs/readme-audit.md` and `docs/dev-notes.md`, re-checking facts affected by recent changes

## Scope

- Edit `README.md`.
- Update documentation only where an audited fact or README decision changed.
- Do not edit source, configurations, checkpoints, datasets, results, or analysis outputs.
- Do not run FL/MIA experiments, external services, or SLURM jobs.
- Do not commit.

## Acceptance criteria

- Add the requested compact table of contents with GitHub-compatible anchors.
- Identify the implementation near the top as a black-box shadow-model MIA based on Shokri et al. (2017), without claiming a novel attack.
- Describe repository-local path defaults and PAULA-specific execution accurately.
- Restructure `Running the project` into the requested developer flow.
- Re-verify commands, paths, Markdown, and numerical claims; inspect the final diff.
- Report remaining TODOs and known limitations.
