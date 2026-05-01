"""Public entrypoint for gradual/slope synthetic experiments."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_synthetic_slope_deltaCmp", run_name="__main__")
