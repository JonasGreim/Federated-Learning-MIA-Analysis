# Project Working Guide

## Scope and language

- This repository is the code base of a finished Master's thesis (federated learning + membership inference attacks). The thesis itself (`masterarbeit.pdf`, German) is graded and final. Do not change research results, experiment configs, checkpoints, or analysis outputs unless a task explicitly says so.
- Treat `experiments_data_analysis/`, `model_checkpoints_target/`, and `experiments_conf/` as read-only by default.
- Write code, code comments, and all project-maintained documentation in English. Explanations to the user may be German.
- Keep `masterarbeit.pdf` out of git (see `.gitignore`). It contains personal data.

## Working method

- Read `README.md`, the current task in `ai/tasks/current/`, and only the relevant files in `docs/` and the code before making changes.
- Use `docs/thesis-facts.md` as the source of truth for numbers and wording of results. Only fall back to the PDF (`pdftotext -layout masterarbeit.pdf`) to check a fact that is missing there, and add it to `docs/thesis-facts.md` with its page number.
- Inspect existing changes (`git status`, `git diff`) before editing and preserve manual user changes.
- If no active task exists, create a concise task specification from the user's request. Do not expand its scope.
- Keep the current task concise and factual: goal, scope, acceptance checks, verified results, open limitations, next step. Move only completed tasks to `ai/tasks/completed/` and record the change in `CHANGELOG.md`.
- Record non-obvious decisions in `docs/decisions.md`.
- Clearly label planned features as planned; do not present them as implemented or tested.
- Before creating or recommending a commit, inspect `.gitignore`, staged and tracked files, and potential secrets (W&B keys, tokens, cluster credentials, personal data).

## Scientific accuracy

- Every number, range, or finding in documentation must appear in `docs/thesis-facts.md` or the thesis PDF. Never estimate, round differently, or infer results.
- Keep the thesis' cautious wording: results are correlational, not causal; the attack is a passive black-box attack on the final global model; findings hold "in the evaluated setup". Avoid absolute words such as "proves", "always", or "every model leaks".
- If code and thesis disagree (for example the type of attack classifier), report the discrepancy instead of silently choosing one. Log it in `docs/decisions.md`.

## Verification

- Every file path, command, and config key mentioned in documentation must exist in the repo. Check with `ls`, `grep`, or by reading the file.
- `pyproject.toml` is the source of truth for Python version and dependencies. `poetry check` is allowed.
- Do NOT run full FL or MIA experiments, SLURM scripts, or anything that needs a GPU or long runtime. Do NOT run `wandb login` or upload anything to Weights & Biases. Prefer static checks (`--help`, imports, reading configs).
- Explicitly report checks that were skipped or could not be verified, and why.

## Sources of truth

- Research design, thesis experiments, numerical research results, and scientific
  conclusions: use `docs/thesis-facts.md` and, when necessary, `masterarbeit.pdf`.
- Current implementation behavior, supported commands, dependencies, config keys,
  defaults, and file paths: verify against the current source code and configuration.
- The old README and AI-generated drafts are not authoritative sources.
- If `docs/thesis-facts.md` conflicts with the final thesis, the thesis takes precedence.
- If the current implementation differs from the thesis implementation, document
  the distinction instead of silently reconciling it.