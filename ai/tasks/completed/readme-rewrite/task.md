# Task: Rewrite README as a portfolio-quality project page

## Goal

Replace the current `README.md` (a mix of setup notes, TODOs, and outdated remarks) with a README that presents the repository as a portfolio project: what it is, what was found, how to run it, how it is structured.

## Inputs

- `docs/thesis-facts.md` (verified results and wording, source of truth for all numbers)
- `masterarbeit.pdf` in repo root (German, final thesis, do not commit) (Scientific facts come from the thesis.
Current technical behavior comes from the repository.
If they differ, report the difference.)
- `ai/tasks/completed/readme-rewrite/draft-claude.md` and `ai/tasks/completed/readme-rewrite/draft-chatgpt.md` (two candidate drafts; use as structure and wording input, not as truth)
- The current `README.md` and the repository itself

## Target structure (in this order)

1. Title, one-line question, short intro, notice/link to the thesis (German; PDF link is a TODO placeholder)
2. Motivation (2-3 sentences)
3. Key findings (4 precise bullets with numbers) + one figure placeholder + heterogeneity example table + short validation results
4. Quick start (install, run the attack on the included example checkpoint, full pipeline)
5. Experimental setup (FL system, rounds table, data splits D1-D4, overfitting indicator)
6. How the attack works (threat model list, pipeline diagram, steps, metrics)
7. Project structure (annotated, only paths that exist)
8. Extending the project and configuration pointers (link to YAML files, do not copy default values)
9. Scope and limitations, short next steps (from thesis chapter 6.3)
10. References

Keep it scannable: roughly 250-330 lines. Do not repeat the thesis title or institution more than once.

## Scope

- Only `README.md` and documentation files (`docs/`, `ai/`, `CHANGELOG.md`).
- Move useful content from the old README (Flower architecture notes, HPC logging, TODO list, "what could be done better") to `docs/dev-notes.md` instead of deleting it. Drop outdated statements.
- Do not change code, configs, or results.

## Out of scope

- Running experiments, creating plots, changing the repo layout, adding CI.
- Adding a LICENSE file (only report whether one exists and what `pyproject.toml` declares).

## Phase 1: audit (no README edits yet)

Produce `docs/readme-audit.md` with a table: claim | source (thesis page or repo file) | status (verified / contradicted / unverifiable) | note. It must cover at least:

- Python version range and main dependencies (`pyproject.toml`)
- License: declared in `pyproject.toml` vs. LICENSE file present
- Attack classifier type in code (`membership_inference_attack/`) vs. thesis (logistic regression per class) vs. old README (RandomForest)
- Number of shadow models and default MIA config values
- Config keys and files mentioned in either draft (`experiments_conf/`, `experiments_conf_types/`)
- Extending instructions: `flower/models/`, `flower/utils/model_factory.py`, `flower/strategies/custom_weighted_fedavg.py`, `split_cifar10_mia.py`, `dataset_splits`
- Checkpoint frequency, default device, whether `wandb login` is mandatory
- How `run_mia_experiment.py` selects the target checkpoint
- Whether `file_name_settings.py` exists and where
- Whether the cluster name (PAULA) and Leipzig University may be named

Stop after Phase 1 and report the audit before continuing.

## Phase 2: write

Write the new README using only verified facts. Mark anything unverifiable as an HTML comment TODO. Then update `CHANGELOG.md` and `docs/decisions.md`.

## Acceptance checks

- [x] All numbers match `docs/thesis-facts.md`
- [x] Every path, command, and config key in the README exists (checked, listed in `docs/readme-audit.md`)
- [x] All code fences are closed; headings render; tables are valid Markdown
- [x] No absolute claims; wording follows `AGENTS.md` (correlational, evaluated setup)
- [x] Useful, current old-README content is preserved in `docs/dev-notes.md`; obsolete claims were dropped
- [x] `masterarbeit.pdf` is not tracked by git
- [x] TODO placeholders exist for: thesis PDF link, one figure

## Verified results

- `README.md` is a 285-line portfolio-oriented overview following the requested section order.
- Scientific numbers and wording were checked against `docs/thesis-facts.md`; bibliography entries were checked against the final thesis (pp. 69-70).
- All linked paths exist, all documented Hydra forms compose successfully with `--cfg job`, and `poetry check` passes.
- Markdown fences are balanced, tables have consistent columns, required TODOs are present, and the thesis title/institution appear once.
- Source, experiment configs, checkpoints, datasets, and analysis outputs are unchanged.

## Open limitations

- Local execution still requires environment-specific edits to `path_settings.py`.
- W&B calls remain unconditional, and the default example attack is a full 100-epoch shadow-model run rather than a smoke test.
- The PAULA scripts remain cluster-specific; stored MIA configs require their matching checkpoint directories.
- The public thesis link, architecture-comparison figure, and repository license file remain TODOs.

## Next step

Add a public thesis link and thesis Figure 5.2 (architecture comparison); add repository topics (`federated-learning`, `membership-inference-attack`, `privacy`, `flower`, `pytorch`); decide on a LICENSE file in a separate task.
