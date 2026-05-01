from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

from .run_synthetic_suddenCmp_tryThm import (
    _summarize_restart_events as summarize_sudden_restart_events,
    _summarize_sudden_result,
    run_one_sudden,
)
from .run_synthetic_slope_deltaCmp import (
    _summarize_slope_result,
    build_phi2_from_theta_star,
    run_one_slope,
)
from .run_synthetic_mixed_thetaCmp import (
    _summarize_mixed_result,
    _summarize_restart_events as summarize_mixed_restart_events,
    run_one_mixed,
)
from .method_names import paper_method_name


RAW_FIELDNAMES = [
    "run_id",
    "raw_relpath",
    "scenario_family",
    "method",
    "family",
    "cpd",
    "seed",
    "num_particles",
    "delta_bpc_lambda",
    "batch_size",
    "total_T",
    "magnitude",
    "seg_len",
    "slope",
    "drift_scale",
    "jump_scale",
    "theta_rmse",
    "theta_crps",
    "y_rmse",
    "y_crps",
    "restart_count",
    "full_restart_count",
    "delta_only_count",
    "gate_refresh_count",
    "false_full_restart_count",
    "post_change_correction_delay",
    "runtime_sec",
]

ERROR_FIELDNAMES = [
    "run_id",
    "scenario_family",
    "method",
    "seed",
    "batch_size",
    "magnitude",
    "seg_len",
    "slope",
    "drift_scale",
    "jump_scale",
    "error_type",
    "error_message",
    "traceback",
]


@dataclass(frozen=True)
class MethodSpec:
    run_key: str
    method: str
    family: str
    cpd: str
    meta: Dict[str, Any]


def _shared_method_name(family: str, cpd: str) -> str:
    return paper_method_name(f"{family}_{cpd}")


def _slug(text: Any) -> str:
    return "".join(ch if str(ch).isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text))


def _append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in fieldnames})


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "run_id" not in df.columns:
        return set()
    return set(df["run_id"].astype(str).tolist())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _restart_count(data: Dict[str, Any]) -> float:
    rm_hist = list(data.get("restart_mode_hist", []))
    if rm_hist:
        return float(sum(1 for v in rm_hist if str(v) != "none"))
    others = list(data.get("others", []))
    return float(sum(1 for item in others if isinstance(item, dict) and bool(item.get("did_restart", False))))


def _base_bocpd_meta(num_particles: int, delta_bpc_lambda: float) -> Dict[str, Any]:
    return dict(
        type="bocpd",
        use_discrepancy=False,
        bocpd_use_discrepancy=True,
        restart_impl="rolled_cusum_260324",
        hybrid_partial_restart=False,
        use_dual_restart=False,
        use_cusum=False,
        particle_delta_mode="shared_gp",
        shared_delta_model="gp",
        delta_update_mode="refit",
        num_particles=int(num_particles),
        delta_bpc_lambda=float(delta_bpc_lambda),
    )


def _controller_meta(mode: str) -> Dict[str, Any]:
    if mode == "None":
        return dict(mode="single_segment", controller_name="none", controller_stat="log_surprise_mean")
    if mode == "BOCPD":
        return dict(mode="restart")
    if mode == "wCUSUM":
        return dict(
            mode="wcusum",
            controller_name="wcusum",
            controller_stat="log_surprise_mean",
            controller_wcusum_warmup_batches=3,
            controller_wcusum_window=4,
            controller_wcusum_threshold=0.25,
            controller_wcusum_kappa=0.25,
            controller_wcusum_sigma_floor=0.25,
        )
    raise ValueError(f"Unsupported controller mode: {mode}")


def build_method_specs(num_particles: int, delta_bpc_lambda: float) -> List[MethodSpec]:
    base = _base_bocpd_meta(num_particles=num_particles, delta_bpc_lambda=delta_bpc_lambda)
    methods: List[MethodSpec] = []

    methods.append(
        MethodSpec(
            run_key="B-BRPC-RRA",
            method="B-BRPC-RRA",
            family="B-BRPC-RRA",
            cpd="BOCPD",
            meta=dict(
                base,
                **_controller_meta("BOCPD"),
                delta_update_mode="refit",
                particle_delta_mode="shared_gp",
                delta_bpc_obs_noise_mode="kernel",
                delta_bpc_predict_add_kernel_noise=True,
            ),
        )
    )

    methods.append(
        MethodSpec(
            run_key="DA",
            method="DA",
            family="DA",
            cpd="None",
            meta=dict(
                type="paper_pf",
                num_particles=int(num_particles),
                paper_pf_sigma_obs_var=0.04,
                paper_pf_move_theta_std=0.15,
                paper_pf_move_logl_std=0.10,
            ),
        )
    )

    methods.append(
        MethodSpec(
            run_key="B-WaldPF",
            method="B-WaldPF",
            family="B-WaldPF",
            cpd="BOCPD",
            meta=dict(
                type="bocpd_paper_pf",
                num_particles=int(num_particles),
                paper_pf_sigma_obs_var=0.04,
                paper_pf_move_theta_std=0.15,
                paper_pf_move_logl_std=0.10,
            ),
        )
    )

    shared_variants = [
        ("Proxy", "online_bpc_proxy_stablemean", ("None", "BOCPD", "wCUSUM")),
        ("Exact", "online_bpc_exact", ("None", "BOCPD", "wCUSUM")),
        ("FixedSupport", "online_bpc_fixedsupport_exact", ("None", "BOCPD", "wCUSUM")),
    ]
    for family, delta_mode, cpds in shared_variants:
        for cpd in cpds:
            meta = dict(
                base,
                **_controller_meta(cpd),
                delta_update_mode=delta_mode,
                delta_bpc_obs_noise_mode="sigma_eps",
                delta_bpc_predict_add_kernel_noise=False,
            )
            methods.append(
                MethodSpec(
                    run_key=_shared_method_name(family, cpd),
                    method=_shared_method_name(family, cpd),
                    family=_shared_method_name(family, cpd) if cpd != "None" else family,
                    cpd=cpd,
                    meta=meta,
                )
            )

    for cpd in ("None", "BOCPD", "wCUSUM"):
        meta = dict(
            base,
            **_controller_meta(cpd),
            particle_delta_mode="particle_gp_fixedsupport_online_bpc_shared_hyper",
            delta_update_mode="online_bpc_fixedsupport_exact",
            delta_bpc_obs_noise_mode="sigma_eps",
            delta_bpc_predict_add_kernel_noise=False,
        )
        methods.append(
            MethodSpec(
                run_key=f"ParticleFixedSupport_{cpd}",
                method=f"ParticleFixedSupport_{cpd}",
                family="ParticleFixedSupport",
                cpd=cpd,
                meta=meta,
            )
        )

    methods.extend(
        [
            MethodSpec(
                run_key="PF-OGP",
                method="PF-OGP",
                family="PF-OGP",
                cpd="None",
                meta=dict(type="pf_ogp"),
            ),
            MethodSpec(
                run_key="BC",
                method="BC",
                family="BC",
                cpd="None",
                meta=dict(type="bc"),
            ),
            MethodSpec(
                run_key="R-BOCPD-PF-OGP",
                method="BOCPD-PF-OGP",
                family="BOCPD-PF-OGP",
                cpd="BOCPD",
                meta=dict(type="bocpd", mode="restart"),
            ),
        ]
    )
    return methods


def _method_registry_df(method_specs: Sequence[MethodSpec]) -> pd.DataFrame:
    rows = []
    for spec in method_specs:
        rows.append(
            {
                "run_key": spec.run_key,
                "method": spec.method,
                "family": spec.family,
                "cpd": spec.cpd,
                "type": spec.meta.get("type", ""),
                "mode": spec.meta.get("mode", ""),
                "delta_update_mode": spec.meta.get("delta_update_mode", ""),
                "particle_delta_mode": spec.meta.get("particle_delta_mode", ""),
                "delta_bpc_lambda": spec.meta.get("delta_bpc_lambda", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _write_progress(path: Path, *, total_runs: int, completed: int, skipped: int, errors: int) -> None:
    payload = {
        "total_runs": int(total_runs),
        "completed_runs": int(completed),
        "skipped_runs": int(skipped),
        "error_runs": int(errors),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _aggregate_outputs(out_dir: Path) -> None:
    raw_path = out_dir / "raw" / "all_runs.csv"
    if not raw_path.exists():
        return
    df = pd.read_csv(raw_path)
    if len(df) == 0:
        return

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "all_runs_copy.csv", index=False)
    if "raw_relpath" in df.columns:
        manifest_cols = [c for c in ["run_id", "scenario_family", "method", "family", "cpd", "seed", "raw_relpath"] if c in df.columns]
        df[manifest_cols].to_csv(summary_dir / "raw_payload_manifest.csv", index=False)

    mean_cols = ["theta_rmse", "theta_crps", "y_rmse", "y_crps", "restart_count", "runtime_sec"]
    extra_mean_cols = [
        "full_restart_count",
        "delta_only_count",
        "gate_refresh_count",
        "false_full_restart_count",
        "post_change_correction_delay",
    ]

    def _grouped(cols: List[str], group_keys: List[str], out_name: str) -> None:
        use_cols = [c for c in cols if c in df.columns]
        if not use_cols:
            return
        grouped = df.groupby(group_keys, dropna=False)[use_cols].agg(["mean", "std"]).reset_index()
        grouped.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).rstrip("_") if isinstance(col, tuple) else str(col)
            for col in grouped.columns
        ]
        grouped.to_csv(summary_dir / out_name, index=False)

    _grouped(mean_cols + extra_mean_cols, ["scenario_family", "method", "family", "cpd"], "scenario_method_mean_summary.csv")
    _grouped(mean_cols + extra_mean_cols, ["scenario_family", "family", "cpd"], "scenario_family_mean_summary.csv")
    _grouped(mean_cols + extra_mean_cols, ["scenario_family", "method"], "scenario_method_only_summary.csv")
    if "magnitude" in df.columns and "seg_len" in df.columns:
        _grouped(mean_cols + extra_mean_cols, ["method", "family", "cpd", "magnitude", "seg_len"], "sudden_config_summary.csv")
    if "slope" in df.columns:
        _grouped(mean_cols + extra_mean_cols, ["method", "family", "cpd", "slope"], "slope_config_summary.csv")
    if "jump_scale" in df.columns:
        _grouped(mean_cols + extra_mean_cols, ["method", "family", "cpd", "drift_scale", "jump_scale"], "mixed_config_summary.csv")


def _sudden_run_id(spec: MethodSpec, seed: int, batch_size: int, magnitude: float, seg_len: int) -> str:
    return f"sudden|{spec.method}|seed={seed}|bs={batch_size}|mag={magnitude:.4f}|seg={seg_len}"


def _slope_run_id(spec: MethodSpec, seed: int, batch_size: int, slope: float) -> str:
    return f"slope|{spec.method}|seed={seed}|bs={batch_size}|slope={slope:.6f}"


def _mixed_run_id(spec: MethodSpec, seed: int, batch_size: int, drift_scale: float, jump_scale: float, total_T: int) -> str:
    return f"mixed|{spec.method}|seed={seed}|bs={batch_size}|drift={drift_scale:.4f}|jump={jump_scale:.3f}|T={total_T}"


def _raw_row_base(
    *,
    run_id: str,
    scenario_family: str,
    spec: MethodSpec,
    seed: int,
    num_particles: int,
    delta_bpc_lambda: float,
    batch_size: int,
    total_T: int,
    magnitude: float = float("nan"),
    seg_len: float = float("nan"),
    slope: float = float("nan"),
    drift_scale: float = float("nan"),
    jump_scale: float = float("nan"),
) -> Dict[str, Any]:
    return dict(
        run_id=run_id,
        scenario_family=scenario_family,
        method=spec.method,
        family=spec.family,
        cpd=spec.cpd,
        seed=int(seed),
        num_particles=int(num_particles),
        delta_bpc_lambda=float(delta_bpc_lambda),
        batch_size=int(batch_size),
        total_T=int(total_T),
        magnitude=magnitude,
        seg_len=seg_len,
        slope=slope,
        drift_scale=drift_scale,
        jump_scale=jump_scale,
    )


def _record_success(raw_path: Path, row: Dict[str, Any]) -> None:
    _append_csv_row(raw_path, RAW_FIELDNAMES, row)


def _record_error(err_path: Path, row: Dict[str, Any]) -> None:
    _append_csv_row(err_path, ERROR_FIELDNAMES, row)


def _save_raw_payload(
    *,
    out_dir: Path,
    scenario_family: str,
    run_id: str,
    spec: MethodSpec,
    config: Dict[str, Any],
    data: Dict[str, Any],
    row: Dict[str, Any],
) -> str:
    rel = Path("raw_runs") / scenario_family / f"{_slug(run_id)}.pt"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = dict(
        run_id=run_id,
        scenario_family=scenario_family,
        method=spec.method,
        family=spec.family,
        cpd=spec.cpd,
        method_meta=dict(spec.meta),
        config=dict(config),
        summary_metrics={k: row.get(k) for k in RAW_FIELDNAMES if k in row and k not in {"run_id", "raw_relpath"}},
    )
    payload.update(data)
    payload["suite_runtime_sec"] = float(row.get("runtime_sec", float("nan")))
    torch.save(payload, path)
    return rel.as_posix()


def _run_sudden(
    *,
    out_dir: Path,
    method_specs: Sequence[MethodSpec],
    seeds: Sequence[int],
    batch_sizes: Sequence[int],
    magnitudes: Sequence[float],
    seg_lens: Sequence[int],
    num_particles: int,
    delta_bpc_lambda: float,
    completed: set[str],
    progress_state: Dict[str, int],
    total_runs: int,
) -> None:
    raw_path = out_dir / "raw" / "all_runs.csv"
    err_path = out_dir / "raw" / "errors.csv"
    progress_path = out_dir / "progress.json"

    for batch_size in batch_sizes:
        for magnitude in magnitudes:
            for seg_len in seg_lens:
                for seed in seeds:
                    for spec in method_specs:
                        run_id = _sudden_run_id(spec, seed, batch_size, magnitude, seg_len)
                        if run_id in completed:
                            progress_state["skipped"] += 1
                            continue
                        print(f"[RUN] {run_id}")
                        t0 = time()
                        try:
                            res, _, _ = run_one_sudden(
                                seg_len_L=seg_len,
                                delta_mag=float(magnitude),
                                methods={spec.run_key: dict(spec.meta)},
                                batch_size=int(batch_size),
                                seed=int(seed),
                                out_dir=str(out_dir / "sudden"),
                            )
                            data = res[spec.run_key]
                            metrics = _summarize_sudden_result(data)
                            restart_stats = summarize_sudden_restart_events(data)
                            row = _raw_row_base(
                                run_id=run_id,
                                scenario_family="sudden",
                                spec=spec,
                                seed=seed,
                                num_particles=num_particles,
                                delta_bpc_lambda=delta_bpc_lambda,
                                batch_size=batch_size,
                                total_T=4 * int(seg_len),
                                magnitude=float(magnitude),
                                seg_len=float(seg_len),
                            )
                            row.update(metrics)
                            row.update(restart_stats)
                            row["restart_count"] = _restart_count(data)
                            row["runtime_sec"] = float(time() - t0)
                            row["raw_relpath"] = _save_raw_payload(
                                out_dir=out_dir,
                                scenario_family="sudden",
                                run_id=run_id,
                                spec=spec,
                                config=dict(
                                    batch_size=int(batch_size),
                                    total_T=4 * int(seg_len),
                                    magnitude=float(magnitude),
                                    seg_len=int(seg_len),
                                    seed=int(seed),
                                    num_particles=int(num_particles),
                                    delta_bpc_lambda=float(delta_bpc_lambda),
                                ),
                                data=data,
                                row=row,
                            )
                            _record_success(raw_path, row)
                            progress_state["completed"] += 1
                            completed.add(run_id)
                        except Exception as exc:
                            _record_error(
                                err_path,
                                dict(
                                    run_id=run_id,
                                    scenario_family="sudden",
                                    method=spec.method,
                                    seed=int(seed),
                                    batch_size=int(batch_size),
                                    magnitude=float(magnitude),
                                    seg_len=int(seg_len),
                                    slope=float("nan"),
                                    drift_scale=float("nan"),
                                    jump_scale=float("nan"),
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    traceback=traceback.format_exc(),
                                ),
                            )
                            progress_state["errors"] += 1
                        _write_progress(
                            progress_path,
                            total_runs=total_runs,
                            completed=progress_state["completed"],
                            skipped=progress_state["skipped"],
                            errors=progress_state["errors"],
                        )


def _run_slope(
    *,
    out_dir: Path,
    method_specs: Sequence[MethodSpec],
    seeds: Sequence[int],
    batch_sizes: Sequence[int],
    slopes: Sequence[float],
    num_particles: int,
    delta_bpc_lambda: float,
    completed: set[str],
    progress_state: Dict[str, int],
    total_runs: int,
) -> None:
    raw_path = out_dir / "raw" / "all_runs.csv"
    err_path = out_dir / "raw" / "errors.csv"
    progress_path = out_dir / "progress.json"

    phi2_grid = np.linspace(3.0, 12.0, 300)
    theta_grid = np.linspace(0.0, 3.0, 600)
    phi2_of_theta, _ = build_phi2_from_theta_star(phi2_grid=phi2_grid, theta_grid=theta_grid)

    for batch_size in batch_sizes:
        for slope in slopes:
            for seed in seeds:
                for spec in method_specs:
                    run_id = _slope_run_id(spec, seed, batch_size, slope)
                    if run_id in completed:
                        progress_state["skipped"] += 1
                        continue
                    print(f"[RUN] {run_id}")
                    t0 = time()
                    try:
                        res, _, _ = run_one_slope(
                            slope=float(slope),
                            methods={spec.run_key: dict(spec.meta)},
                            total_T=600,
                            batch_size=int(batch_size),
                            seed=int(seed),
                            phi2_of_theta=phi2_of_theta,
                            mode=1,
                        )
                        data = res[spec.run_key]
                        metrics = _summarize_slope_result(data)
                        row = _raw_row_base(
                            run_id=run_id,
                            scenario_family="slope",
                            spec=spec,
                            seed=seed,
                            num_particles=num_particles,
                            delta_bpc_lambda=delta_bpc_lambda,
                            batch_size=batch_size,
                            total_T=600,
                            slope=float(slope),
                        )
                        row.update(metrics)
                        row["restart_count"] = _restart_count(data)
                        row["full_restart_count"] = float("nan")
                        row["delta_only_count"] = float("nan")
                        row["gate_refresh_count"] = float("nan")
                        row["false_full_restart_count"] = float("nan")
                        row["post_change_correction_delay"] = float("nan")
                        row["runtime_sec"] = float(time() - t0)
                        row["raw_relpath"] = _save_raw_payload(
                            out_dir=out_dir,
                            scenario_family="slope",
                            run_id=run_id,
                            spec=spec,
                            config=dict(
                                batch_size=int(batch_size),
                                total_T=600,
                                slope=float(slope),
                                seed=int(seed),
                                num_particles=int(num_particles),
                                delta_bpc_lambda=float(delta_bpc_lambda),
                            ),
                            data=data,
                            row=row,
                        )
                        _record_success(raw_path, row)
                        progress_state["completed"] += 1
                        completed.add(run_id)
                    except Exception as exc:
                        _record_error(
                            err_path,
                            dict(
                                run_id=run_id,
                                scenario_family="slope",
                                method=spec.method,
                                seed=int(seed),
                                batch_size=int(batch_size),
                                magnitude=float("nan"),
                                seg_len=float("nan"),
                                slope=float(slope),
                                drift_scale=float("nan"),
                                jump_scale=float("nan"),
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                traceback=traceback.format_exc(),
                            ),
                        )
                        progress_state["errors"] += 1
                    _write_progress(
                        progress_path,
                        total_runs=total_runs,
                        completed=progress_state["completed"],
                        skipped=progress_state["skipped"],
                        errors=progress_state["errors"],
                    )


def _run_mixed(
    *,
    out_dir: Path,
    method_specs: Sequence[MethodSpec],
    seeds: Sequence[int],
    batch_sizes: Sequence[int],
    drift_scales: Sequence[float],
    jump_scales: Sequence[float],
    total_T: int,
    num_particles: int,
    delta_bpc_lambda: float,
    completed: set[str],
    progress_state: Dict[str, int],
    total_runs: int,
) -> None:
    raw_path = out_dir / "raw" / "all_runs.csv"
    err_path = out_dir / "raw" / "errors.csv"
    progress_path = out_dir / "progress.json"

    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=np.linspace(3.0, 12.0, 300),
        theta_grid=np.linspace(0.0, 3.0, 600),
    )

    for batch_size in batch_sizes:
        for drift_scale in drift_scales:
            for jump_scale in jump_scales:
                for seed in seeds:
                    for spec in method_specs:
                        run_id = _mixed_run_id(spec, seed, batch_size, drift_scale, jump_scale, total_T)
                        if run_id in completed:
                            progress_state["skipped"] += 1
                            continue
                        print(f"[RUN] {run_id}")
                        t0 = time()
                        try:
                            res, _, _, _, _ = run_one_mixed(
                                drift_scale=float(drift_scale),
                                jump_scale=float(jump_scale),
                                methods={spec.run_key: dict(spec.meta)},
                                batch_size=int(batch_size),
                                seed=int(seed),
                                total_T=int(total_T),
                                phi2_of_theta=phi2_of_theta,
                                num_particles=int(num_particles),
                            )
                            data = res[spec.run_key]
                            metrics = _summarize_mixed_result(data)
                            restart_stats = summarize_mixed_restart_events(data)
                            row = _raw_row_base(
                                run_id=run_id,
                                scenario_family="mixed",
                                spec=spec,
                                seed=seed,
                                num_particles=num_particles,
                                delta_bpc_lambda=delta_bpc_lambda,
                                batch_size=batch_size,
                                total_T=total_T,
                                drift_scale=float(drift_scale),
                                jump_scale=float(jump_scale),
                            )
                            row.update(metrics)
                            row.update(restart_stats)
                            row["restart_count"] = _restart_count(data)
                            row["runtime_sec"] = float(time() - t0)
                            row["raw_relpath"] = _save_raw_payload(
                                out_dir=out_dir,
                                scenario_family="mixed",
                                run_id=run_id,
                                spec=spec,
                                config=dict(
                                    batch_size=int(batch_size),
                                    total_T=int(total_T),
                                    drift_scale=float(drift_scale),
                                    jump_scale=float(jump_scale),
                                    seed=int(seed),
                                    num_particles=int(num_particles),
                                    delta_bpc_lambda=float(delta_bpc_lambda),
                                ),
                                data=data,
                                row=row,
                            )
                            _record_success(raw_path, row)
                            progress_state["completed"] += 1
                            completed.add(run_id)
                        except Exception as exc:
                            _record_error(
                                err_path,
                                dict(
                                    run_id=run_id,
                                    scenario_family="mixed",
                                    method=spec.method,
                                    seed=int(seed),
                                    batch_size=int(batch_size),
                                    magnitude=float("nan"),
                                    seg_len=float("nan"),
                                    slope=float("nan"),
                                    drift_scale=float(drift_scale),
                                    jump_scale=float(jump_scale),
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    traceback=traceback.format_exc(),
                                ),
                            )
                            progress_state["errors"] += 1
                        _write_progress(
                            progress_path,
                            total_runs=total_runs,
                            completed=progress_state["completed"],
                            skipped=progress_state["skipped"],
                            errors=progress_state["errors"],
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "sudden", "slope", "mixed"])
    parser.add_argument("--seed_count", type=int, default=25)
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--delta_bpc_lambda", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--sudden_magnitudes", type=float, nargs="*", default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--sudden_seg_lens", type=int, nargs="*", default=[80, 120, 200])
    parser.add_argument("--sudden_batch_sizes", type=int, nargs="*", default=[20])
    parser.add_argument("--slope_values", type=float, nargs="*", default=[0.0005, 0.001, 0.0015, 0.002, 0.0025])
    parser.add_argument("--slope_batch_sizes", type=int, nargs="*", default=[20])
    parser.add_argument("--mixed_batch_sizes", type=int, nargs="*", default=[20])
    parser.add_argument("--mixed_drift_scales", type=float, nargs="*", default=[0.008])
    parser.add_argument("--mixed_jump_scales", type=float, nargs="*", default=[0.28, 0.38, 0.58])
    parser.add_argument("--mixed_total_T", type=int, default=600)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    method_specs = build_method_specs(num_particles=args.num_particles, delta_bpc_lambda=args.delta_bpc_lambda)
    if args.methods:
        keep = {paper_method_name(m) for m in args.methods}
        method_specs = [spec for spec in method_specs if spec.method in keep or spec.run_key in keep]
        missing = sorted(keep - {spec.method for spec in method_specs} - {spec.run_key for spec in method_specs})
        if missing:
            raise ValueError(f"Unknown methods requested: {missing}")
    _method_registry_df(method_specs).to_csv(out_dir / "summary" / "method_registry.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed_count": int(args.seed_count),
                "num_particles": int(args.num_particles),
                "delta_bpc_lambda": float(args.delta_bpc_lambda),
                "sudden_magnitudes": list(args.sudden_magnitudes),
                "sudden_seg_lens": list(args.sudden_seg_lens),
                "sudden_batch_sizes": list(args.sudden_batch_sizes),
                "slope_values": list(args.slope_values),
                "slope_batch_sizes": list(args.slope_batch_sizes),
                "mixed_batch_sizes": list(args.mixed_batch_sizes),
                "mixed_drift_scales": list(args.mixed_drift_scales),
                "mixed_jump_scales": list(args.mixed_jump_scales),
                "mixed_total_T": int(args.mixed_total_T),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seeds = list(range(int(args.seed_count)))
    total_runs = 0
    if args.scenario in {"all", "sudden"}:
        total_runs += len(args.sudden_batch_sizes) * len(args.sudden_magnitudes) * len(args.sudden_seg_lens) * len(seeds) * len(method_specs)
    if args.scenario in {"all", "slope"}:
        total_runs += len(args.slope_batch_sizes) * len(args.slope_values) * len(seeds) * len(method_specs)
    if args.scenario in {"all", "mixed"}:
        total_runs += len(args.mixed_batch_sizes) * len(args.mixed_drift_scales) * len(args.mixed_jump_scales) * len(seeds) * len(method_specs)

    completed = _load_completed(out_dir / "raw" / "all_runs.csv") if args.resume else set()
    progress_state = {"completed": len(completed), "skipped": 0, "errors": 0}
    _write_progress(out_dir / "progress.json", total_runs=total_runs, completed=progress_state["completed"], skipped=0, errors=0)

    if args.scenario in {"all", "sudden"}:
        _run_sudden(
            out_dir=out_dir,
            method_specs=method_specs,
            seeds=seeds,
            batch_sizes=args.sudden_batch_sizes,
            magnitudes=args.sudden_magnitudes,
            seg_lens=args.sudden_seg_lens,
            num_particles=args.num_particles,
            delta_bpc_lambda=args.delta_bpc_lambda,
            completed=completed,
            progress_state=progress_state,
            total_runs=total_runs,
        )
        _aggregate_outputs(out_dir)

    if args.scenario in {"all", "slope"}:
        _run_slope(
            out_dir=out_dir,
            method_specs=method_specs,
            seeds=seeds,
            batch_sizes=args.slope_batch_sizes,
            slopes=args.slope_values,
            num_particles=args.num_particles,
            delta_bpc_lambda=args.delta_bpc_lambda,
            completed=completed,
            progress_state=progress_state,
            total_runs=total_runs,
        )
        _aggregate_outputs(out_dir)

    if args.scenario in {"all", "mixed"}:
        _run_mixed(
            out_dir=out_dir,
            method_specs=method_specs,
            seeds=seeds,
            batch_sizes=args.mixed_batch_sizes,
            drift_scales=args.mixed_drift_scales,
            jump_scales=args.mixed_jump_scales,
            total_T=args.mixed_total_T,
            num_particles=args.num_particles,
            delta_bpc_lambda=args.delta_bpc_lambda,
            completed=completed,
            progress_state=progress_state,
            total_runs=total_runs,
        )
        _aggregate_outputs(out_dir)

    _aggregate_outputs(out_dir)
    (out_dir / "RUN_COMPLETE.txt").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
