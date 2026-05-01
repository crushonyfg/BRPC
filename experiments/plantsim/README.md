# PlantSim Experiments

Primary runner:

```bash
conda run -n jumpGP python -m calib.run_plantSim_v3_std --help
```

Example:

```bash
conda run -n jumpGP python -m calib.run_plantSim_v3_std --csv physical_data.csv --out_dir figs/plantSim/v3_std --modes 0 1 2
```

Keep `physical_data.csv`, `factory_aggregated.npz`, and emulator bundles at the
repository root unless the runner defaults are updated together.
