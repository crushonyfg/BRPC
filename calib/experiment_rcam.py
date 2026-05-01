"""Public entrypoint for one-dimensional RCAM wind-jump diagnostics."""

import runpy


if __name__ == "__main__":
    runpy.run_module("calib.run_rcam_bocpd_pf", run_name="__main__")
