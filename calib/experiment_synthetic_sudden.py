"""Public entrypoint for abrupt synthetic changepoint experiments."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_synthetic_suddenCmp_tryThm", run_name="__main__")
