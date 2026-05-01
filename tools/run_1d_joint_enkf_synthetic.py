from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.joint_enkf import JointEnKF1D, JointEnKFConfig
from calib.online_calibrator import crps_gaussian
from calib.run_synthetic_slope_deltaCmp import (
    ThetaDrivenSlopeDataStream,
    build_phi2_from_theta_star,
    computer_model_config2_np,
    oracle_theta,
)
from calib.run_synthetic_suddenCmp_tryThm import (
    SuddenChangeDataStream,
    build_phi_segments_centered,
    physical_system,
)
from calib.run_synthetic_mixed_thetaCmp import MixedThetaDataStream


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _crps_mean(mu: np.ndarray, var: np.ndarray, y: np.ndarray) -> float:
    return float(
        crps_gaussian(
            torch.tensor(mu, dtype=torch.float64),
            torch.tensor(np.clip(var, 1e-12, None), dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
        ).mean().item()
    )


def _make_phi2_of_theta():
    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=np.linspace(3.0, 12.0, 300),
        theta_grid=np.linspace(0.0, 3.0, 600),
    )
    return phi2_of_theta


def _make_stream(scenario: str, seed: int, batch_size: int, noise_sd: float):
    if scenario == "sudden":
        seg_len = 120
        cp_times = [seg_len, 2 * seg_len, 3 * seg_len]
        return SuddenChangeDataStream(
            total_T=4 * seg_len,
            batch_size=batch_size,
            noise_sd=noise_sd,
            cp_times=cp_times,
            phi_segments=build_phi_segments_centered(delta=2.0, center=7.5),
            seed=seed,
        )
    phi2_of_theta = _make_phi2_of_theta()
    if scenario == "slope":
        return ThetaDrivenSlopeDataStream(
            total_T=600,
            batch_size=batch_size,
            noise_sd=noise_sd,
            theta0=1.5,
            theta_slope=0.0015,
            phi2_of_theta=phi2_of_theta,
            seed=seed,
        )
    if scenario == "mixed":
        return MixedThetaDataStream(
            total_T=600,
            batch_size=batch_size,
            noise_sd=noise_sd,
            phi2_of_theta=phi2_of_theta,
            drift_scale=0.008,
            jump_scale=0.38,
            theta_noise_sd=0.015,
            seed=seed,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def run_one(
    scenario: str,
    seed: int,
    out_dir: Path,
    batch_size: int,
    noise_sd: float,
    n_ensemble: int,
    theta_rw_sd: float,
    beta_rw_sd: float,
    covariance_inflation: float,
) -> Dict[str, float]:
    stream = _make_stream(scenario, seed=seed, batch_size=batch_size, noise_sd=noise_sd)
    enkf = JointEnKF1D(
        computer_model_config2_np,
        JointEnKFConfig(
            n_ensemble=int(n_ensemble),
            sigma_obs=float(noise_sd),
            theta_rw_sd=float(theta_rw_sd),
            beta_rw_sd=float(beta_rw_sd),
            covariance_inflation=float(covariance_inflation),
            seed=int(seed) + 9101,
        ),
    )

    theta_grid = np.linspace(0.0, 3.0, 600)
    theta_hist: List[float] = []
    theta_var_hist: List[float] = []
    oracle_hist: List[float] = []
    phi_hist: List[np.ndarray] = []
    rmse_hist: List[float] = []
    crps_hist: List[float] = []
    X_batches: List[np.ndarray] = []
    Y_batches: List[np.ndarray] = []
    y_noiseless_batches: List[np.ndarray] = []
    pred_mu_batches: List[np.ndarray] = []
    pred_var_batches: List[np.ndarray] = []

    t0 = time()
    batch_idx = 0
    while stream.t < stream.T:
        Xb, Yb = stream.next()
        X_np = Xb.detach().cpu().numpy()
        Y_np = Yb.detach().cpu().numpy().reshape(-1)
        phi_t = np.asarray(stream.phi_history[-1], dtype=float).copy()
        y0_np = np.asarray(physical_system(X_np, phi_t), dtype=float).reshape(-1)

        if batch_idx > 0:
            mu_np, var_np = enkf.predict(X_np)
            rmse_hist.append(float(np.sqrt(np.mean((mu_np - Y_np) ** 2))))
            crps_hist.append(_crps_mean(mu_np, var_np, Y_np))
            pred_mu_batches.append(np.asarray(mu_np, dtype=float).reshape(-1).copy())
            pred_var_batches.append(np.asarray(var_np, dtype=float).reshape(-1).copy())
        else:
            pred_mu_batches.append(np.full_like(Y_np, np.nan, dtype=float))
            pred_var_batches.append(np.full_like(Y_np, np.nan, dtype=float))

        X_batches.append(X_np.copy())
        Y_batches.append(Y_np.copy())
        y_noiseless_batches.append(y0_np.copy())

        enkf.update_batch(X_np, Y_np)
        theta_hist.append(enkf.mean_theta())
        theta_var_hist.append(max(enkf.var_theta(), 1e-12))
        phi_hist.append(phi_t.copy())
        oracle_hist.append(float(oracle_theta(phi_t, theta_grid)))
        batch_idx += 1

    runtime_sec = float(time() - t0)
    theta_arr = np.asarray(theta_hist, dtype=float)
    oracle_arr = np.asarray(oracle_hist, dtype=float)
    theta_var_arr = np.asarray(theta_var_hist, dtype=float)
    theta_rmse = float(np.sqrt(np.mean((theta_arr - oracle_arr) ** 2)))
    theta_crps = _crps_mean(theta_arr, theta_var_arr, oracle_arr)

    raw_dir = out_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_seed{seed}_bs{batch_size}_JointEnKF"
    raw_path = raw_dir / f"{tag}.pt"
    payload = dict(
        method="JointEnKF",
        scenario=scenario,
        seed=int(seed),
        batch_size=int(batch_size),
        theta=theta_arr,
        theta_oracle=oracle_arr,
        theta_var=theta_var_arr,
        phi_hist=np.asarray(phi_hist, dtype=object),
        rmse=np.asarray(rmse_hist, dtype=float),
        crps_hist=np.asarray(crps_hist, dtype=float),
        X_batches=np.asarray(X_batches, dtype=object),
        Y_batches=np.asarray(Y_batches, dtype=object),
        y_noiseless_batches=np.asarray(y_noiseless_batches, dtype=object),
        pred_mu_batches=np.asarray(pred_mu_batches, dtype=object),
        pred_var_batches=np.asarray(pred_var_batches, dtype=object),
        elapsed_sec=runtime_sec,
    )
    if hasattr(stream, "cp_times"):
        payload["cp_times"] = list(getattr(stream, "cp_times"))
    if hasattr(stream, "cp_batches"):
        payload["cp_batches"] = list(getattr(stream, "cp_batches"))
    torch.save(payload, raw_path)

    plot_dir = out_dir / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 4.5))
    plt.plot(oracle_arr, "k--", lw=2.0, label="oracle theta*")
    plt.plot(theta_arr, color="tab:blue", lw=1.8, label="JointEnKF")
    cp_times = list(payload.get("cp_times", []))
    for cp in cp_times:
        plt.axvline(int(cp) // int(batch_size), color="tab:red", ls="--", alpha=0.35)
    plt.xlabel("batch index")
    plt.ylabel("theta")
    plt.title(f"Joint EnKF theta tracking ({scenario}, seed={seed})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{tag}_theta_tracking.png", dpi=220)
    plt.close()

    pd.DataFrame(
        dict(
            batch_idx=np.arange(len(theta_arr)),
            theta=theta_arr,
            theta_oracle=oracle_arr,
            theta_var=theta_var_arr,
        )
    ).to_csv(plot_dir / f"{tag}_theta_tracking.csv", index=False)

    return dict(
        scenario=scenario,
        method="JointEnKF",
        seed=int(seed),
        batch_size=int(batch_size),
        n_ensemble=int(n_ensemble),
        theta_rw_sd=float(theta_rw_sd),
        beta_rw_sd=float(beta_rw_sd),
        covariance_inflation=float(covariance_inflation),
        theta_rmse=theta_rmse,
        theta_crps=theta_crps,
        y_rmse=_finite_mean(rmse_hist),
        y_crps=_finite_mean(crps_hist),
        runtime_sec=runtime_sec,
        raw_relpath=str(raw_path.relative_to(out_dir)).replace("\\", "/"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/joint_enkf_1d_synthetic")
    parser.add_argument("--scenarios", nargs="+", choices=["slope", "sudden", "mixed", "all"], default=["all"])
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--noise_sd", type=float, default=0.2)
    parser.add_argument("--n_ensemble", type=int, default=512)
    parser.add_argument("--theta_rw_sd", type=float, default=0.035)
    parser.add_argument("--beta_rw_sd", type=float, default=0.015)
    parser.add_argument("--covariance_inflation", type=float, default=1.02)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["slope", "sudden", "mixed"] if "all" in args.scenarios else list(args.scenarios)

    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
            print(f"[joint-enkf] scenario={scenario} seed={seed}")
            rows.append(
                run_one(
                    scenario=scenario,
                    seed=seed,
                    out_dir=out_dir,
                    batch_size=int(args.batch_size),
                    noise_sd=float(args.noise_sd),
                    n_ensemble=int(args.n_ensemble),
                    theta_rw_sd=float(args.theta_rw_sd),
                    beta_rw_sd=float(args.beta_rw_sd),
                    covariance_inflation=float(args.covariance_inflation),
                )
            )
            summary_dir = out_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            run_df = pd.DataFrame(rows)
            run_df.to_csv(summary_dir / "run_level.csv", index=False)
            agg = (
                run_df.groupby(["scenario", "method", "batch_size", "n_ensemble", "theta_rw_sd", "beta_rw_sd", "covariance_inflation"], as_index=False)
                .agg(
                    theta_rmse_mean=("theta_rmse", "mean"),
                    theta_rmse_std=("theta_rmse", "std"),
                    theta_crps_mean=("theta_crps", "mean"),
                    y_rmse_mean=("y_rmse", "mean"),
                    y_rmse_std=("y_rmse", "std"),
                    y_crps_mean=("y_crps", "mean"),
                    runtime_sec_mean=("runtime_sec", "mean"),
                )
            )
            agg.to_csv(summary_dir / "scenario_summary.csv", index=False)
            with (summary_dir / "manifest.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "method": "JointEnKF",
                        "notes": "Joint EnKF baseline over [theta, discrepancy-basis coefficients] for representative 1D synthetic scenarios.",
                        "scenarios": scenarios,
                        "seeds": sorted(run_df["seed"].unique().tolist()),
                    },
                    fh,
                    indent=2,
                )


if __name__ == "__main__":
    main()
