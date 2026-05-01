"""Public entrypoint for PlantSim / factory-data experiments."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_plantSim_v3_std", run_name="__main__")
