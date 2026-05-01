"""Public entrypoint for high-dimensional projected BRPC diagnostics."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_synthetic_highdim_projected_diag", run_name="__main__")
