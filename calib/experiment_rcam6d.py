"""Public entrypoint for six-dimensional RCAM calibration diagnostics."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_rcam6d_bocpd_pf", run_name="__main__")
