# AGENTS.md

PlantSim experiments use the factory/physical data runners.

- Main runner: `calib.experiment_plantsim`.
- Prefer passing data paths explicitly, for example `--csv physical_data.csv` and `--npz factory_aggregated.npz`.
- Keep generated model bundles and figures out of source-controlled package folders.
