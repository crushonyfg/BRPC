import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.joint_enkf import JointEnKFConfig, JointEnKFGeneric
from calib.online_calibrator import crps_gaussian
from calib.run_plantSim_v3_std import (
    JumpPlan,
    StreamClass,
    _default_sigma_eps_s,
    _gaussian_crps_mean_raw,
    _parse_max_batches_by_mode,
    _raw_pred_from_standardized,
    batch_X_base_to_s,
    batch_y_to_s,
    batches,
    init_pipeline,
)


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


def _make_stream(mode: int, seed: int, data_dir: str | None, csv_path: str | None):
    if int(mode) == 2:
        jp = JumpPlan(
            max_jumps=5,
            min_gap_theta=500.0,
            min_interval=180,
            max_interval=320,
            min_jump_span=40,
            seed=int(seed),
        )
        return StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
    return StreamClass(int(mode), folder=data_dir, csv_path=csv_path)


def run_one(
    mode: int,
    seed: int,
    out_dir: Path,
    batch_size: int,
    max_batches: int | None,
    data_dir: str | None,
    csv_path: str | None,
    npz_path: str | None,
    model_save_path: str,
    n_ensemble: int,
    theta_rw_sd: float,
    beta_rw_sd: float,
    covariance_inflation: float,
    beta_init_sd: float,
    beta_damping: float,
    num_basis: int,
    basis_lengthscale: float,
    sigma_obs_s: float,
) -> Dict[str, float]:
    if data_dir is None and csv_path is None:
        data_dir = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"
    gt, nn_std, _, a_s, b_s = init_pipeline(npz_path=npz_path, model_save_path=model_save_path)

    def sim_func_std_np(x_np: np.ndarray, theta_np: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_np, dtype=np.float64)
        th_arr = np.asarray(theta_np, dtype=np.float64).reshape(-1, 1)
        if x_arr.shape[0] != th_arr.shape[0]:
            raise ValueError(f"Expected one-to-one standardized inputs, got X={x_arr.shape}, theta={th_arr.shape}")
        x_full = np.concatenate([x_arr, th_arr], axis=1)
        return np.asarray(nn_std.predict_y_s_from_Xfull_s(x_full), dtype=float).reshape(-1)

    def theta_uniform(n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(float(a_s), float(b_s), size=(int(n), 1))

    enkf = JointEnKFGeneric(
        sim_func_std_np,
        JointEnKFConfig(
            n_ensemble=int(n_ensemble),
            theta_lo=float(a_s),
            theta_hi=float(b_s),
            sigma_obs=float(sigma_obs_s),
            beta_init_sd=float(beta_init_sd),
            theta_rw_sd=float(theta_rw_sd),
            beta_rw_sd=float(beta_rw_sd),
            beta_damping=float(beta_damping),
            num_basis=int(num_basis),
            basis_lengthscale=float(basis_lengthscale),
            covariance_inflation=float(covariance_inflation),
            seed=int(seed) + 22031,
        ),
        x_dim=5,
        theta_init_sampler=theta_uniform,
    )

    stream = _make_stream(mode=mode, seed=seed, data_dir=data_dir, csv_path=csv_path)
    theta_hist: List[float] = []
    theta_var_hist: List[float] = []
    gt_theta_hist: List[float] = []
    rmse_hist: List[float] = []
    y_crps_hist: List[float] = []
    y_true_hist: List[np.ndarray] = []
    y_pred_hist: List[np.ndarray] = []
    y_var_hist: List[np.ndarray] = []

    t0 = time()
    for batch_idx, (xb_raw, yb_raw, thb_raw) in enumerate(batches(stream, int(batch_size), max_batches=max_batches)):
        x_s = batch_X_base_to_s(gt, xb_raw).detach().cpu().numpy()
        y_s = batch_y_to_s(gt, yb_raw).detach().cpu().numpy().reshape(-1)
        y_raw = np.asarray(yb_raw, dtype=float).reshape(-1)

        if batch_idx > 0:
            mu_s, var_s = enkf.predict(x_s)
            mu_raw, var_raw = _raw_pred_from_standardized(gt, mu_s, var_s)
            rmse_hist.append(float(np.sqrt(np.mean((mu_raw - y_raw) ** 2))))
            y_crps_hist.append(_gaussian_crps_mean_raw(gt, mu_s, var_s, y_raw))
            y_true_hist.append(y_raw.copy())
            y_pred_hist.append(mu_raw.copy())
            y_var_hist.append(var_raw.copy())

        enkf.update_batch(x_s, y_s)
        theta_s_mean = enkf.mean_theta()
        theta_s_var = max(enkf.var_theta(), 1e-12)
        theta_hist.append(float(gt.theta_s_to_raw(np.asarray([theta_s_mean]))[0]))
        theta_var_hist.append(float(theta_s_var * (gt.theta_sd ** 2)))
        gt_theta_hist.append(float(np.mean(thb_raw)))

    runtime_sec = float(time() - t0)
    theta_arr = np.asarray(theta_hist, dtype=float)
    gt_theta_arr = np.asarray(gt_theta_hist, dtype=float)
    theta_var_arr = np.asarray(theta_var_hist, dtype=float)
    theta_rmse = float(np.sqrt(np.mean((theta_arr - gt_theta_arr) ** 2)))
    theta_crps = _crps_mean(theta_arr, theta_var_arr, gt_theta_arr)

    raw_dir = out_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tag = f"mode{mode}_seed{seed}_bs{batch_size}_JointEnKF"
    raw_path = raw_dir / f"{tag}.pt"
    torch.save(
        dict(
            method="JointEnKF",
            dataset="plantsim",
            mode=int(mode),
            seed=int(seed),
            batch_size=int(batch_size),
            theta=np.asarray(theta_hist, dtype=float),
            theta_oracle=np.asarray(gt_theta_hist, dtype=float),
            theta_var=np.asarray(theta_var_hist, dtype=float),
            rmse=np.asarray(rmse_hist, dtype=float),
            crps_hist=np.asarray(y_crps_hist, dtype=float),
            y_true_hist=np.asarray(y_true_hist, dtype=object),
            y_pred_hist=np.asarray(y_pred_hist, dtype=object),
            y_var_hist=np.asarray(y_var_hist, dtype=object),
            elapsed_sec=runtime_sec,
        ),
        raw_path,
    )

    plot_dir = out_dir / "theta_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 4.5))
    plt.plot(gt_theta_arr, "k--", lw=2.0, label="physical theta")
    plt.plot(theta_arr, color="tab:blue", lw=1.8, label="JointEnKF")
    plt.xlabel("batch index")
    plt.ylabel("theta raw")
    plt.title(f"PlantSim Joint EnKF theta tracking (mode={mode}, seed={seed})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{tag}_theta_tracking.png", dpi=220)
    plt.close()
    pd.DataFrame(
        dict(batch_idx=np.arange(len(theta_arr)), theta=theta_arr, theta_oracle=gt_theta_arr, theta_var=theta_var_arr)
    ).to_csv(plot_dir / f"{tag}_theta_tracking.csv", index=False)

    return dict(
        dataset="plantsim",
        scenario=f"mode{mode}",
        method="JointEnKF",
        seed=int(seed),
        batch_size=int(batch_size),
        max_batches=float(max_batches) if max_batches is not None else np.nan,
        n_ensemble=int(n_ensemble),
        theta_rw_sd=float(theta_rw_sd),
        beta_rw_sd=float(beta_rw_sd),
        covariance_inflation=float(covariance_inflation),
        beta_init_sd=float(beta_init_sd),
        beta_damping=float(beta_damping),
        num_basis=int(num_basis),
        basis_lengthscale=float(basis_lengthscale),
        theta_rmse=theta_rmse,
        theta_crps=theta_crps,
        y_rmse=_finite_mean(rmse_hist),
        y_crps=_finite_mean(y_crps_hist),
        runtime_sec=runtime_sec,
        raw_relpath=str(raw_path.relative_to(out_dir)).replace("\\", "/"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/plantsim_joint_enkf")
    parser.add_argument("--modes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max-batches-by-mode", nargs="*", default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default="nn_std.bundle.joblib")
    parser.add_argument("--n_ensemble", type=int, default=512)
    parser.add_argument("--theta_rw_sd", type=float, default=0.035)
    parser.add_argument("--beta_rw_sd", type=float, default=0.04)
    parser.add_argument("--covariance_inflation", type=float, default=1.0)
    parser.add_argument("--beta_init_sd", type=float, default=0.25)
    parser.add_argument("--beta_damping", type=float, default=0.995)
    parser.add_argument("--num_basis", type=int, default=32)
    parser.add_argument("--basis_lengthscale", type=float, default=1.0)
    parser.add_argument("--sigma_obs_s", type=float, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_caps = _parse_max_batches_by_mode(args.max_batches_by_mode)
    sigma_obs_s = _default_sigma_eps_s() if args.sigma_obs_s is None else float(args.sigma_obs_s)

    rows: List[Dict[str, float]] = []
    for mode in args.modes:
        mode_cap = mode_caps.get(int(mode), args.max_batches)
        for seed in range(int(args.seed_offset), int(args.seed_offset) + int(args.seed_count)):
            print(f"[plantsim-enkf] mode={mode} seed={seed} max_batches={mode_cap}")
            rows.append(
                run_one(
                    mode=int(mode),
                    seed=int(seed),
                    out_dir=out_dir,
                    batch_size=int(args.batch_size),
                    max_batches=mode_cap,
                    data_dir=args.data_dir,
                    csv_path=args.csv,
                    npz_path=args.npz_path,
                    model_save_path=args.model_save_path,
                    n_ensemble=int(args.n_ensemble),
                    theta_rw_sd=float(args.theta_rw_sd),
                    beta_rw_sd=float(args.beta_rw_sd),
                    covariance_inflation=float(args.covariance_inflation),
                    beta_init_sd=float(args.beta_init_sd),
                    beta_damping=float(args.beta_damping),
                    num_basis=int(args.num_basis),
                    basis_lengthscale=float(args.basis_lengthscale),
                    sigma_obs_s=sigma_obs_s,
                )
            )
            summary_dir = out_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            run_df = pd.DataFrame(rows)
            run_df.to_csv(summary_dir / "run_level.csv", index=False)
            agg = (
                run_df.groupby(
                    [
                        "scenario",
                        "method",
                        "batch_size",
                        "n_ensemble",
                        "theta_rw_sd",
                        "beta_rw_sd",
                        "covariance_inflation",
                    ],
                    as_index=False,
                )
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
                    dict(
                        method="JointEnKF",
                        notes="Joint EnKF baseline over standardized PlantSim theta and RFF discrepancy-basis coefficients.",
                        modes=[int(m) for m in args.modes],
                        seeds=sorted(run_df["seed"].unique().tolist()),
                        sigma_obs_s=sigma_obs_s,
                    ),
                    fh,
                    indent=2,
                )


if __name__ == "__main__":
    main()
