# Task: Final README editorial polish

## Goal

Perform a targeted final editorial pass on `README.md` without changing its established structure.

## Scope

- Re-check the README against current path settings, Hydra configuration, project metadata, implementation behavior, and `docs/thesis-facts.md`.
- Improve clarity, consistency, grammar, and public-facing readability.
- Do not modify source code, experiment configuration, checkpoints, datasets, analysis outputs, or research conclusions.
- Do not commit.

## Acceptance checks

- Preserve existing README sections and order.
- Verify every changed technical claim, path, command, and numerical result.
- Check Markdown links, heading anchors, code fences, and whitespace.
- Report skipped end-to-end checks and remaining TODOs.

## Verified results

- Polished wording and terminology without restructuring the README.
- Confirmed repository-local path defaults, ignored generated artifacts, Hydra defaults and overrides, checkpoint selection, and CPU commands against the current repository.
- Confirmed research numbers and conclusions against `docs/thesis-facts.md`.
- Static Hydra composition, Poetry metadata, links, anchors, fences, and scoped whitespace checks passed.

## Open limitations

- Full FL/MIA experiments, dataset downloads, W&B initialization, GPU workloads, and SLURM jobs were intentionally not run.
- The public thesis link and architecture-comparison figure remain README TODOs.

## Next step

Task complete; archive it under `ai/tasks/completed/`. No commit was created.
