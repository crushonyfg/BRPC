from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rbf_cov_np(x1: np.ndarray, x2: np.ndarray, lengthscale: float, variance: float) -> np.ndarray:
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    ls = max(float(lengthscale), 1e-8)
    diff = (a[:, None, :] - b[None, :, :]) / ls
    dist2 = np.sum(diff * diff, axis=-1)
    return float(variance) * np.exp(-0.5 * dist2)


def _stabilize(a: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = 0.5 * (a + a.T)
    if a.shape[0] == 0:
        return a
    scale = max(float(np.mean(np.diag(a))) if a.size else 1.0, 1.0)
    return a + float(jitter) * scale * np.eye(a.shape[0])


def _posterior_cov_from_prior(prior_cov: np.ndarray, obs_noise: float, eta_delta: float = 1.0) -> np.ndarray:
    n = prior_cov.shape[0]
    # Tempering the likelihood by eta_delta is equivalent to using
    # R_eff = R / eta_delta.  If eta_delta = 1 / lambda_delta, then
    # R_eff = lambda_delta * R.
    eta = max(float(eta_delta), 1e-12)
    obs_noise_eff = max(float(obs_noise) / eta, 1e-10)
    s = _stabilize(prior_cov + obs_noise_eff * np.eye(n))
    solve = np.linalg.solve(s, prior_cov.T).T
    post = prior_cov - solve @ prior_cov
    return _stabilize(post)


def _max_generalized_eig(c: np.ndarray, m: np.ndarray) -> float:
    c = _stabilize(c, jitter=1e-10)
    m = _stabilize(m, jitter=1e-8)
    try:
        return float(np.max(eigvalsh(c, m)))
    except Exception:
        return float(np.max(np.real(np.linalg.eigvals(np.linalg.solve(m, c)))))


def _gamma_for_transition(
    x_prev: np.ndarray,
    p_prev: np.ndarray,
    x_cur: np.ndarray,
    *,
    lengthscale: float,
    variance: float,
    obs_noise: float,
    eta_delta: float,
) -> tuple[float, float, np.ndarray]:
    k_prev = _stabilize(_rbf_cov_np(x_prev, x_prev, lengthscale, variance), jitter=1e-10)
    k_prev_cur = _rbf_cov_np(x_prev, x_cur, lengthscale, variance)
    a = np.linalg.solve(k_prev, k_prev_cur).T
    k_cur = _stabilize(_rbf_cov_np(x_cur, x_cur, lengthscale, variance), jitter=1e-10)
    schur = k_cur - k_prev_cur.T @ np.linalg.solve(k_prev, k_prev_cur)
    prior_cov = _stabilize(schur + a @ p_prev @ a.T)
    p_cur = _posterior_cov_from_prior(prior_cov, obs_noise, eta_delta=eta_delta)

    m_prev = _stabilize(np.linalg.inv(p_prev), jitter=1e-8)
    m_prior = _stabilize(np.linalg.inv(prior_cov), jitter=1e-8)
    m_post = _stabilize(np.linalg.inv(p_cur), jitter=1e-8)
    gamma_prior = _max_generalized_eig(a.T @ m_prior @ a, m_prev)
    gamma_post = _max_generalized_eig(a.T @ m_post @ a, m_prev)
    return gamma_prior, gamma_post, p_cur


def gamma_series_from_batches(
    x_batches: Sequence[np.ndarray],
    *,
    lengthscale: float,
    variance: float,
    obs_noise: float,
    eta_delta: float,
) -> List[Dict[str, float]]:
    batches = [np.asarray(x, dtype=float).reshape(np.asarray(x).shape[0], -1) for x in x_batches if len(x) > 0]
    if len(batches) < 2:
        return []
    k0 = _stabilize(_rbf_cov_np(batches[0], batches[0], lengthscale, variance), jitter=1e-10)
    p_prev = _posterior_cov_from_prior(k0, obs_noise, eta_delta=eta_delta)
    out: List[Dict[str, float]] = []
    for x_prev, x_cur in zip(batches[:-1], batches[1:]):
        gamma_prior, gamma_post, p_prev = _gamma_for_transition(
            x_prev,
            p_prev,
            x_cur,
            lengthscale=lengthscale,
            variance=variance,
            obs_noise=obs_noise,
            eta_delta=eta_delta,
        )
        if np.isfinite(gamma_prior) and np.isfinite(gamma_post):
            out.append(
                dict(
                    gamma_prior=float(gamma_prior),
                    gamma_post=float(gamma_post),
                    gamma_post_minus_prior=float(gamma_post - gamma_prior),
                )
            )
    return out


def _iter_synthetic_payloads(raw_dir: Path) -> Iterable[tuple[Path, str, dict]]:
    for path in sorted(raw_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        methods = payload.get("methods", payload if isinstance(payload, dict) else {})
        if not isinstance(methods, dict):
            continue
        for method, data in methods.items():
            if isinstance(data, dict) and "X_batches" in data:
                yield path, str(method), data


def collect_synthetic(
    raw_dir: Path,
    lengthscale: float,
    variance: float,
    obs_noise: float,
    eta_delta: float,
    keep_methods: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seen_sources: set[str] = set()
    for path, method, data in _iter_synthetic_payloads(raw_dir):
        source_key = str(path)
        if not keep_methods and source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        gammas = gamma_series_from_batches(
            list(data["X_batches"]),
            lengthscale=lengthscale,
            variance=variance,
            obs_noise=obs_noise,
            eta_delta=eta_delta,
        )
        scenario = "unknown"
        name = path.name
        if name.startswith("slope"):
            scenario = "slope"
        elif name.startswith("sudden"):
            scenario = "sudden"
        elif name.startswith("mixed"):
            scenario = "mixed"
        for idx, gamma_row in enumerate(gammas):
            rows.append(
                dict(
                    dataset="synthetic_1d",
                    source_path=str(path),
                    scenario=scenario,
                    method=method if keep_methods else "exact_gp_replay",
                    batch_idx=idx + 1,
                    **gamma_row,
                )
            )
    return pd.DataFrame(rows)


def _plant_batches(mode: int, seed: int, batch_size: int, max_batches: int | None) -> List[np.ndarray]:
    from calib.run_plantSim_v3_std import (
        JumpPlan,
        StreamClass,
        batch_X_base_to_s,
        batches,
        init_pipeline,
    )

    gt, _, _, _, _ = init_pipeline()
    if mode == 2:
        stream = StreamClass(
            0,
            folder="C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3",
            csv_path=None,
            jump_plan=JumpPlan(max_jumps=5, min_gap_theta=500.0, min_interval=180, seed=int(seed)),
        )
    else:
        stream = StreamClass(mode, folder="C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3", csv_path=None)
    out: List[np.ndarray] = []
    for xb, _, _ in batches(stream, batch_size, max_batches=max_batches):
        out.append(batch_X_base_to_s(gt, xb).detach().cpu().numpy())
    return out


def collect_plant(
    modes: Sequence[int],
    seeds: Sequence[int],
    batch_size: int,
    max_batches: int | None,
    lengthscale: float,
    variance: float,
    obs_noise: float,
    eta_delta: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for mode in modes:
        for seed in seeds:
            x_batches = _plant_batches(int(mode), int(seed), int(batch_size), max_batches)
            gammas = gamma_series_from_batches(
                x_batches,
                lengthscale=lengthscale,
                variance=variance,
                obs_noise=obs_noise,
                eta_delta=eta_delta,
            )
            for idx, gamma_row in enumerate(gammas):
                rows.append(
                    dict(
                        dataset="plantsim",
                        source_path=f"mode={mode},seed={seed}",
                        scenario=f"mode{mode}",
                        method="transport_only",
                        batch_idx=idx + 1,
                        **gamma_row,
                    )
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(["dataset", "scenario", "method"], as_index=False).size().rename(columns={"size": "n"})
    for col in ["gamma_prior", "gamma_post", "gamma_post_minus_prior"]:
        stats = (
            df.groupby(["dataset", "scenario", "method"])[col]
            .agg(
                **{
                    f"{col}_median": "median",
                    f"{col}_p90": lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.90)),
                    f"{col}_p95": lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.95)),
                    f"{col}_max": "max",
                    f"{col}_frac_le_1": lambda x: float(np.mean(np.asarray(x, dtype=float) <= 1.0)),
                    f"{col}_frac_le_1p1": lambda x: float(np.mean(np.asarray(x, dtype=float) <= 1.1)),
                }
            )
            .reset_index()
        )
        summary = summary.merge(stats, on=["dataset", "scenario", "method"], how="left")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/transport_gamma_diagnostic")
    parser.add_argument("--synthetic_raw_dir", type=str, default="")
    parser.add_argument("--include_plant", action="store_true", default=False)
    parser.add_argument("--plant_modes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--plant_seed_count", type=int, default=5)
    parser.add_argument("--plant_batch_size", type=int, default=4)
    parser.add_argument("--plant_max_batches", type=int, default=250)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=0.01)
    parser.add_argument("--obs_noise", type=float, default=0.0025)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--eta_delta", type=float, default=None)
    parser.add_argument(
        "--keep_synthetic_methods",
        action="store_true",
        default=False,
        help="Keep duplicate synthetic method labels. By default, each saved design is replayed once as exact_gp_replay.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eta_delta = float(args.eta_delta) if args.eta_delta is not None else 1.0 / max(float(args.lambda_delta), 1e-12)
    frames = []
    if args.synthetic_raw_dir:
        frames.append(
            collect_synthetic(
                Path(args.synthetic_raw_dir),
                lengthscale=float(args.lengthscale),
                variance=float(args.variance),
                obs_noise=float(args.obs_noise),
                eta_delta=eta_delta,
                keep_methods=bool(args.keep_synthetic_methods),
            )
        )
    if args.include_plant:
        frames.append(
            collect_plant(
                modes=args.plant_modes,
                seeds=list(range(int(args.plant_seed_count))),
                batch_size=int(args.plant_batch_size),
                max_batches=int(args.plant_max_batches),
                lengthscale=float(args.lengthscale),
                variance=float(args.variance),
                obs_noise=float(args.obs_noise),
                eta_delta=eta_delta,
            )
        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(out_dir / "gamma_run_level.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out_dir / "gamma_summary.csv", index=False)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "formula_prior": "lambda_max(M_prev^{-1/2} A^T (P_t^-)^{-1} A M_prev^{-1/2})",
                "formula_post": "lambda_max(M_prev^{-1/2} A^T C_t^{-1} A M_prev^{-1/2})",
                "lengthscale": float(args.lengthscale),
                "variance": float(args.variance),
                "obs_noise": float(args.obs_noise),
                "lambda_delta": float(args.lambda_delta),
                "eta_delta": float(eta_delta),
                "notes": "gamma_prior is the pre-update transport diagnostic closest to Assumption 1; gamma_post is the current-batch posterior-sharpening diagnostic.",
            },
            fh,
            indent=2,
        )
    print(summ)


if __name__ == "__main__":
    main()
