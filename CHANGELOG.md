# Changelog

## Unreleased

### Documentation

- Replaced the provisional README with a verified portfolio-oriented project overview.
- Documented the evaluated findings, experimental design, MIA pipeline, current setup requirements, Hydra usage, checkpoint handoff, outputs, extension points, and limitations.
- Added `docs/readme-audit.md` with claim-level source verification.
- Established `docs/` as the canonical location for stable documentation and moved the thesis fact sheet and decision log there.
- Added `docs/dev-notes.md` for detailed Flower architecture, PAULA/SLURM logging, output behavior, known implementation constraints, and planned engineering improvements.
- Preserved TODO placeholders for a public thesis PDF and the architecture-comparison figure.
- Added README navigation, clarified that the implemented attack is the black-box shadow-model method from Shokri et al. (2017), and reorganized the execution guide around configuration, FL, MIA, the FL-to-MIA handoff, outputs, and PAULA.
- Updated path documentation after `path_settings.py` switched from PAULA `/work` roots to repository-local defaults.
- Completed a final editorial pass for terminology, range formatting, generated-artifact guidance, client split wording, and concise public-facing prose.

No source code, experiment configuration, checkpoints, datasets, or analysis outputs were changed.
