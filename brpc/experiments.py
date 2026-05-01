"""
Paper-facing experiment registry.

This module intentionally stores command metadata instead of wrapping the
runner internals. The original runners remain import-compatible under
``calib`` and can be launched with ``python -m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ExperimentEntry:
    module: str
    purpose: str
    example_args: Tuple[str, ...]

    @property
    def command(self) -> str:
        args = " ".join(self.example_args)
        return f"conda run -n jumpGP python -m {self.module} {args}".strip()


EXPERIMENTS: Dict[str, ExperimentEntry] = {
    "synthetic_sudden": ExperimentEntry(
        module="calib.run_synthetic_suddenCmp_tryThm",
        purpose="Synthetic abrupt changepoint comparison for BRPC variants.",
        example_args=("--profile", "main", "--out_dir", "figs/sudden_main"),
    ),
    "synthetic_slope": ExperimentEntry(
        module="calib.run_synthetic_slope_deltaCmp",
        purpose="Synthetic smooth/slope drift comparison for discrepancy variants.",
        example_args=("--profile", "main", "--out_dir", "figs/slope_main"),
    ),
    "synthetic_mixed": ExperimentEntry(
        module="calib.run_synthetic_mixed_thetaCmp",
        purpose="Mixed theta trajectory benchmark and ablation runner.",
        example_args=("--profile", "main", "--out_dir", "figs/mixed_main"),
    ),
    "synthetic_cpd_suite": ExperimentEntry(
        module="calib.run_synthetic_cpd_suite",
        purpose="Configurable CPD suite over saved raw payloads and method sets.",
        example_args=("--help",),
    ),
    "plantsim": ExperimentEntry(
        module="calib.run_plantSim_v3_std",
        purpose="PlantSim / factory data experiment runner.",
        example_args=("--help",),
    ),
    "highdim_projected": ExperimentEntry(
        module="calib.run_synthetic_highdim_projected_diag",
        purpose="High-dimensional projected diagnostic experiment.",
        example_args=("--help",),
    ),
}


def list_commands() -> str:
    lines = []
    for name, entry in EXPERIMENTS.items():
        lines.append(f"{name}: {entry.command}")
    return "\n".join(lines)


def main() -> None:
    print(list_commands())


if __name__ == "__main__":
    main()
