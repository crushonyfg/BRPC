"""
Public compatibility package for the Jump Gaussian Process / BRPC code.

The implementation modules currently live in the legacy ``calib`` package.
Importing through ``brpc`` gives new users a paper-facing namespace while
preserving all existing ``python -m calib...`` experiment entrypoints.
"""

from calib import (  # noqa: F401
    BOCPDConfig,
    CalibrationConfig,
    DeterministicSimulator,
    GPEmulator,
    ModelConfig,
    OnlineBayesCalibrator,
    PFConfig,
    SyntheticDataStream,
    SyntheticGeneratorConfig,
)

__version__ = "0.1.0"

__all__ = [
    "BOCPDConfig",
    "CalibrationConfig",
    "DeterministicSimulator",
    "GPEmulator",
    "ModelConfig",
    "OnlineBayesCalibrator",
    "PFConfig",
    "SyntheticDataStream",
    "SyntheticGeneratorConfig",
]
