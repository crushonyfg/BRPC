"""Public entrypoint for mixed theta-trajectory synthetic experiments."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_synthetic_mixed_thetaCmp", run_name="__main__")
