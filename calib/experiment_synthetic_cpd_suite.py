"""Public entrypoint for the configurable synthetic CPD benchmark suite."""

import runpy

if __name__ == "__main__":
    runpy.run_module("calib.run_synthetic_cpd_suite", run_name="__main__")
