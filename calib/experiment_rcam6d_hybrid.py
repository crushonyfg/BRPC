"""Public entrypoint for rolled-CUSUM RCAM hybrid diagnostics."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_rcam6d_hybrid_rolled", run_name="__main__")
