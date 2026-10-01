# Project documentation decisions

## 2026-09-30 - README rewrite

### Audience and structure

- The README is an English portfolio page for developers, researchers, and recruiters.
- Results appear before implementation details.
- The thesis is identified once as German and associated with Leipzig University.
- PAULA may be named when describing the evaluated infrastructure.
- Stable project documentation lives under `docs/`; task material and draft inputs remain under `ai/` until the task is completed.

### Sources of truth

- `docs/thesis-facts.md` controls research design, numerical results, and scientific wording.
- Current source, configuration, and `pyproject.toml` control implementation claims.
- `docs/readme-audit.md` records verification and discrepancies.
- The old README and AI drafts provide structure or wording only.

### Scientific wording

- Use the thesis wording that architecture was the strongest observed factor.
- Describe relationships as correlational and limited to the evaluated setup.
- Describe the attack as passive, black-box, class-specific logistic regression against the final global model.
- Do not repeat the old README's RandomForest claim.

### Running the repository

- Do not claim that a fresh clone runs out of the box.
- Document the active repository-local path defaults, generated dataset prerequisite, W&B integration, and manual FL-to-MIA checkpoint handoff.
- Keep the PAULA/SLURM workflow separate and identify its optional `/work` layout and other environment-specific assumptions.
- Use only Hydra selectors verified during the audit, including subgroup-qualified experiment paths.
- The included example checkpoint is described only as a tracked round-10 checkpoint that loads as `simple_model`; no untracked training provenance is claimed.
- Source/config portability defects are not fixed by this documentation task. Execution-relevant requirements stay in the README; lower-level bugs and operational details stay in `docs/readme-audit.md` and `docs/dev-notes.md`.

### Maintenance boundaries

- Config defaults are linked rather than duplicated in the extension section.
- Useful architectural, HPC logging, and maintenance notes from the old README are preserved in `docs/dev-notes.md`; obsolete commands and historical claims are dropped.
- No License section is added while the repository has no `LICENSE` file, even though `pyproject.toml` declares Apache-2.0.
- `masterarbeit.pdf` remains ignored and untracked because it contains personal data. The README keeps a TODO for a deliberate public thesis link.

## 2026-09-30 - Targeted README polish

- Add a compact table of contents for the README's major sections and execution flow.
- Identify the implementation near the top as a black-box shadow-model MIA based on Shokri et al. (2017), without presenting it as a new attack method.
- Prefer `poetry run python ...` in executable examples so commands use the locked project environment without assuming an activated shell.
- Present repository-local execution first and keep PAULA/SLURM guidance explicitly environment-specific.
