from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import math

import torch

from .configs import BOCPDConfig, ModelConfig, PFConfig
from .emulator import Emulator
from .expert_delta import ExpertDeltaFitter
from .likelihood import predictive_stats
from .restart_bocpd_rolled_cusum_260324_gpytorch import BOCPD as RestartBOCPD, Expert


class SingleSegmentController(RestartBOCPD):
    """
    Single-segment controller for online-BPC variants.

    It reuses the existing PF and discrepancy update helpers from RestartBOCPD,
    but replaces the outer multi-expert BOCPD controller with either:
    - no changepoint detection (`controller_name="none"`)
    - a Shekhar-Ramdas style confidence-sequence emptiness test on
      pre-update batch surprise (`controller_name="sr_cs"`)
    - a window-limited CUSUM / GLR-style score-shift detector
      (`controller_name="wcusum"`)
    """

    def __init__(
        self,
        config: BOCPDConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        delta_fitter: Optional[ExpertDeltaFitter] = None,
        on_restart: Optional[Callable] = None,
        notify_on_restart: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config,
            device=device,
            dtype=dtype,
            delta_fitter=delta_fitter,
            on_restart=on_restart,
            notify_on_restart=notify_on_restart,
            **kwargs,
        )
        name = str(getattr(config, "controller_name", "none")).lower()
        self.controller_name = name if name in {"none", "sr_cs", "wcusum"} else "none"
        self.controller_stat = str(getattr(config, "controller_stat", "surprise_mean")).lower()
        self._sr_active: List[Dict[str, float]] = []
        self._wcusum_scores: List[float] = []
        self._segment_batch_count: int = 0

    def _reset_detector_state(self) -> None:
        self._sr_active = []
        self._wcusum_scores = []
        self._segment_batch_count = 0

    def _ensure_single_expert(
        self,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        dx: int,
    ) -> Expert:
        if len(self.experts) == 0:
            self.experts = [self._spawn_new_expert(model_cfg, pf_cfg, prior_sampler, dx, log_mass=0.0)]
        elif len(self.experts) > 1:
            best = max(self.experts, key=lambda e: float(e.log_mass))
            self.experts = [best]
        self.experts[0].log_mass = 0.0
        return self.experts[0]

    def _controller_clip(self, raw: float) -> float:
        lo = float(getattr(self.config, "controller_cs_clip_low", 0.0))
        hi = float(getattr(self.config, "controller_cs_clip_high", 20.0))
        return min(max(float(raw), lo), hi)

    def _pre_update_calibration_gap_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> float:
        if len(self.experts) == 0:
            return float("nan")
        e = self.experts[0]
        ps = e.pf.particles
        mu_eta, var_eta = emulator.predict(X_batch, ps.theta)
        use_discrepancy = bool(model_cfg.bocpd_use_discrepancy)
        if use_discrepancy and e.delta_state is not None:
            if hasattr(e.delta_state, "predict_for_particles"):
                mu_delta, var_delta = e.delta_state.predict_for_particles(
                    X_batch,
                    ps.theta,
                    emulator=emulator,
                    rho=model_cfg.rho,
                )
            else:
                mu_delta, var_delta = e.delta_state.predict(X_batch)
            if mu_eta.dim() == 3 and mu_delta.dim() == 1:
                dy = mu_eta.shape[-1]
                mu_delta = mu_delta[:, None].expand(-1, dy)
                var_delta = var_delta[:, None].expand(-1, dy)
        else:
            if mu_eta.dim() == 3:
                dy = mu_eta.shape[-1]
                mu_delta = torch.zeros(X_batch.shape[0], dy, dtype=mu_eta.dtype, device=mu_eta.device)
                var_delta = torch.zeros(X_batch.shape[0], dy, dtype=mu_eta.dtype, device=mu_eta.device)
            else:
                mu_delta = torch.zeros(X_batch.shape[0], dtype=mu_eta.dtype, device=mu_eta.device)
                var_delta = torch.zeros(X_batch.shape[0], dtype=mu_eta.dtype, device=mu_eta.device)
        mu_tot, var_tot = predictive_stats(model_cfg.rho, mu_eta, var_eta, mu_delta, var_delta, model_cfg.sigma_eps)
        w = ps.weights()
        if mu_tot.dim() == 2:
            w2 = w.view(1, -1)
            mu_mix = (mu_tot * w2).sum(dim=1)
            sec_mix = ((var_tot + mu_tot.square()) * w2).sum(dim=1)
            var_mix = (sec_mix - mu_mix.square()).clamp_min(1e-12)
            y_obs = Y_batch.view(-1).to(mu_mix)
        else:
            w3 = w.view(1, -1, 1)
            mu_mix = (mu_tot * w3).sum(dim=1)
            sec_mix = ((var_tot + mu_tot.square()) * w3).sum(dim=1)
            var_mix = (sec_mix - mu_mix.square()).clamp_min(1e-12)
            y_obs = Y_batch if Y_batch.dim() > 1 else Y_batch[:, None]
            y_obs = y_obs.to(mu_mix)
        z2 = ((y_obs - mu_mix).square() / var_mix).clamp(max=1e6)
        return float(z2.mean() - 1.0)

    def _pre_update_score_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> Dict[str, float]:
        out = dict(
            raw=float("nan"),
            clipped=float("nan"),
            mean_logpred=float("nan"),
            calibration_gap=float("nan"),
        )
        if len(self.experts) == 0:
            return out
        stat = self.controller_stat
        if stat in {"surprise_mean", "log_surprise_mean"}:
            vals = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
            if not vals:
                return out
            mean_logpred = float(vals[0])
            raw_surprise = float(-mean_logpred)
            if not math.isfinite(raw_surprise):
                return out
            raw = raw_surprise if stat == "surprise_mean" else math.log1p(max(raw_surprise, 0.0))
            if not math.isfinite(raw):
                return out
            out.update(raw=raw, clipped=self._controller_clip(raw), mean_logpred=mean_logpred)
            return out
        if stat == "calibration_gap_mean":
            gap = self._pre_update_calibration_gap_batch(X_batch, Y_batch, emulator, model_cfg)
            if not math.isfinite(gap):
                return out
            out.update(raw=gap, clipped=self._controller_clip(gap), calibration_gap=gap)
            return out
        return out

    def _sr_update_and_check(self, score: float) -> Dict[str, Any]:
        alpha = max(float(getattr(self.config, "controller_cs_alpha", 0.01)), 1e-12)
        min_len = max(int(getattr(self.config, "controller_cs_min_len", 2)), 1)
        warmup = max(int(getattr(self.config, "controller_cs_warmup_batches", 2)), 1)
        max_active = max(int(getattr(self.config, "controller_cs_max_active", 64)), 1)
        lo = float(getattr(self.config, "controller_cs_clip_low", 0.0))
        hi = float(getattr(self.config, "controller_cs_clip_high", 20.0))
        span = max(hi - lo, 1e-8)

        updated: List[Dict[str, float]] = []
        for state in self._sr_active:
            updated.append(
                dict(
                    start_batch=int(state["start_batch"]),
                    n=int(state["n"]) + 1,
                    score_sum=float(state["score_sum"]) + float(score),
                )
            )
        updated.append(dict(start_batch=int(self._segment_batch_count), n=1, score_sum=float(score)))
        if len(updated) > max_active:
            updated = updated[-max_active:]
        self._sr_active = updated

        intervals = []
        for state in self._sr_active:
            n = int(state["n"])
            if n < min_len:
                continue
            mean = float(state["score_sum"]) / float(n)
            radius = span * math.sqrt(math.log(2.0 / alpha) / (2.0 * float(n)))
            intervals.append((mean - radius, mean + radius))

        lower = max((iv[0] for iv in intervals), default=float("nan"))
        upper = min((iv[1] for iv in intervals), default=float("nan"))
        total_seen = int(self._segment_batch_count) + 1
        alarm = bool(
            len(intervals) >= 2
            and total_seen >= warmup
            and math.isfinite(lower)
            and math.isfinite(upper)
            and lower > upper
        )
        return {
            "alarm": alarm,
            "intersection_lower": lower,
            "intersection_upper": upper,
            "num_active": int(len(self._sr_active)),
            "num_intervals": int(len(intervals)),
            "segment_batches_seen": total_seen,
        }

    def _wcusum_update_and_check(self, score: float) -> Dict[str, Any]:
        warmup = max(int(getattr(self.config, "controller_wcusum_warmup_batches", 3)), 1)
        window = max(int(getattr(self.config, "controller_wcusum_window", 4)), 1)
        threshold = float(getattr(self.config, "controller_wcusum_threshold", 2.0))
        kappa = float(getattr(self.config, "controller_wcusum_kappa", 0.25))
        sigma_floor = max(float(getattr(self.config, "controller_wcusum_sigma_floor", 0.25)), 1e-8)

        self._wcusum_scores.append(float(score))
        total_seen = len(self._wcusum_scores)
        if total_seen <= warmup + 1:
            return {
                "alarm": False,
                "stat": 0.0,
                "baseline_mean": float(sum(self._wcusum_scores) / max(total_seen, 1)),
                "baseline_sigma": sigma_floor,
                "segment_batches_seen": total_seen,
            }

        best_stat = 0.0
        best_mu0 = float('nan')
        best_sigma0 = float('nan')
        max_m = min(window, total_seen - warmup)
        for m in range(1, max_m + 1):
            ref = self._wcusum_scores[:-m]
            if len(ref) < warmup:
                continue
            mu0 = float(sum(ref) / len(ref))
            if len(ref) >= 2:
                var0 = sum((s - mu0) ** 2 for s in ref) / float(len(ref) - 1)
                sigma0 = max(math.sqrt(max(var0, 0.0)), sigma_floor)
            else:
                sigma0 = sigma_floor
            recent = self._wcusum_scores[-m:]
            recent_mean = float(sum(recent) / len(recent))
            stat = math.sqrt(float(m)) * max(((recent_mean - mu0) / sigma0) - kappa, 0.0)
            if stat > best_stat:
                best_stat = stat
                best_mu0 = mu0
                best_sigma0 = sigma0
        alarm = bool(best_stat > threshold)
        return {
            "alarm": alarm,
            "stat": best_stat,
            "baseline_mean": best_mu0,
            "baseline_sigma": best_sigma0,
            "segment_batches_seen": total_seen,
        }

    def _call_restart_hook(self, t_now: int) -> None:
        if self.on_restart is not None and self.notify_on_restart:
            self.on_restart(int(t_now), int(self.restart_start_time), None, 0, 0.0, 1.0)

    def _experts_debug(self, t_now: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, e in enumerate(self.experts):
            try:
                theta_mean = self._expert_theta_mean(e).detach().cpu().tolist()
            except Exception:
                theta_mean = None
            out.append(
                {
                    "index": idx,
                    "run_length": int(e.run_length),
                    "start_time": int(t_now - e.run_length),
                    "log_mass": float(e.log_mass),
                    "mass": float(math.exp(e.log_mass)),
                    "theta_mean": theta_mean,
                    "log_ump": None,
                }
            )
        return out

    def update(self, x_t: torch.Tensor, y_t: torch.Tensor, emulator: Emulator, model_cfg: ModelConfig, pf_cfg: PFConfig, prior_sampler: Callable[[int], torch.Tensor], verbose: bool = False):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)
        return self.update_batch(x_t, y_t, emulator, model_cfg, pf_cfg, prior_sampler, verbose=verbose)

    def update_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        del verbose
        batch_size = int(X_batch.shape[0])
        dx = int(X_batch.shape[1])
        t_old = int(self.t)

        self._ensure_single_expert(model_cfg, pf_cfg, prior_sampler, dx)
        score_info = self._pre_update_score_batch(X_batch, Y_batch, emulator, model_cfg)
        sr_diag = {
            "alarm": False,
            "intersection_lower": float("nan"),
            "intersection_upper": float("nan"),
            "num_active": 0,
            "num_intervals": 0,
            "segment_batches_seen": int(self._segment_batch_count),
        }
        wcusum_diag = {
            "alarm": False,
            "stat": float("nan"),
            "baseline_mean": float("nan"),
            "baseline_sigma": float("nan"),
            "segment_batches_seen": int(self._segment_batch_count),
        }

        did_restart = False
        restart_mode = "none"
        if self.controller_name == "sr_cs" and math.isfinite(score_info["clipped"]):
            sr_diag = self._sr_update_and_check(float(score_info["clipped"]))
            if sr_diag["alarm"]:
                did_restart = True
                restart_mode = "sr_cs"
                self.restart_start_time = int(t_old)
                self.experts = [self._spawn_new_expert(model_cfg, pf_cfg, prior_sampler, dx, log_mass=0.0)]
                self._reset_detector_state()
                self._call_restart_hook(t_old)
        elif self.controller_name == "wcusum" and math.isfinite(score_info["raw"]):
            wcusum_diag = self._wcusum_update_and_check(float(score_info["raw"]))
            if wcusum_diag["alarm"]:
                did_restart = True
                restart_mode = "wcusum"
                self.restart_start_time = int(t_old)
                self.experts = [self._spawn_new_expert(model_cfg, pf_cfg, prior_sampler, dx, log_mass=0.0)]
                self._reset_detector_state()
                self._call_restart_hook(t_old)

        e = self._ensure_single_expert(model_cfg, pf_cfg, prior_sampler, dx)
        self._append_hist_batch(e, X_batch, Y_batch, self.config.max_run_length)
        diag = e.pf.step_batch(
            X_batch,
            Y_batch,
            emulator,
            e.delta_state,
            model_cfg.rho,
            model_cfg.sigma_eps,
            grad_info=False,
            use_discrepancy=model_cfg.use_discrepancy,
        )
        self._update_delta_after_batch(e, X_batch, Y_batch, emulator, model_cfg, diag)
        e.run_length += batch_size
        e.log_mass = 0.0
        self.t = int(t_old + batch_size)
        self.experts = [e]
        self._segment_batch_count += 1

        t_now = int(self.t)
        return {
            "delta_ll_pair": None,
            "log_odds_mass": None,
            "h_log": float("nan"),
            "log_ump_anchor": score_info["mean_logpred"],
            "log_ump_cand": None,
            "log_umps": [score_info["mean_logpred"]] if math.isfinite(score_info["mean_logpred"]) else [],
            "log_Z": 0.0,
            "entropy": 0.0,
            "experts_debug": self._experts_debug(t_now),
            "experts_log_mass": [0.0],
            "pf_diags": [diag],
            "did_restart": did_restart,
            "restart_mode": restart_mode,
            "restart_start_time": int(self.restart_start_time),
            "s_star": None,
            "anchor_rl": int(e.run_length),
            "cand_rl": 0 if did_restart else None,
            "controller_name": self.controller_name,
            "controller_stat": self.controller_stat,
            "controller_pre_update_score": score_info["raw"],
            "controller_pre_update_score_clipped": score_info["clipped"],
            "controller_pre_update_surprise": score_info["raw"] if self.controller_stat in {"surprise_mean", "log_surprise_mean"} else float("nan"),
            "controller_pre_update_surprise_clipped": score_info["clipped"] if self.controller_stat in {"surprise_mean", "log_surprise_mean"} else float("nan"),
            "controller_pre_update_mean_logpred": score_info["mean_logpred"],
            "controller_pre_update_calibration_gap": score_info["calibration_gap"],
            "controller_cs_lower": sr_diag["intersection_lower"],
            "controller_cs_upper": sr_diag["intersection_upper"],
            "controller_num_active": sr_diag["num_active"],
            "controller_num_intervals": sr_diag["num_intervals"],
            "controller_wcusum_stat": wcusum_diag["stat"],
            "controller_wcusum_baseline_mean": wcusum_diag["baseline_mean"],
            "controller_wcusum_baseline_sigma": wcusum_diag["baseline_sigma"],
            "controller_segment_batches_seen": sr_diag["segment_batches_seen"] if self.controller_name == "sr_cs" else wcusum_diag["segment_batches_seen"],
        }
