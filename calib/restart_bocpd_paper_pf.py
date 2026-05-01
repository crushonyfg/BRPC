from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .configs import BOCPDConfig
from .paper_pf_digital_twin import WardPaperPFConfig, WardPaperParticleFilter


@dataclass
class ExpertPaperPF:
    run_length: int
    pf: WardPaperParticleFilter
    log_mass: float


def _logsumexp_np(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vmax = float(np.max(values))
    return float(vmax + np.log(np.exp(values - vmax).sum()))


class BOCPD_PaperPF:
    def __init__(
        self,
        config: BOCPDConfig,
        pf_cfg: WardPaperPFConfig,
        sim_func_np: Callable,
        on_restart: Optional[Callable] = None,
        notify_on_restart: bool = True,
    ):
        self.config = config
        self.pf_cfg = pf_cfg
        self.sim_func_np = sim_func_np
        self.on_restart = on_restart
        self.notify_on_restart = notify_on_restart

        self.experts: List[ExpertPaperPF] = []
        self.t: int = 0
        self.restart_start_time: int = 0
        self._last_restart_t: int = -(10 ** 9)
        self.prev_max_ump: float = 0.0
        self._spawn_count: int = 0

        self.restart_margin = float(getattr(config, "restart_margin", 0.05))
        self.restart_cooldown = int(getattr(config, "restart_cooldown", 10))
        self.restart_criteria = str(getattr(config, "restart_criteria", "rank_change"))

    def _spawn_new_expert(self, log_mass: float) -> ExpertPaperPF:
        seed = int(getattr(self.pf_cfg, "seed", 0)) + 7919 * self._spawn_count
        self._spawn_count += 1
        cfg = replace(self.pf_cfg, seed=seed)
        pf = WardPaperParticleFilter(self.sim_func_np, cfg)
        return ExpertPaperPF(run_length=0, pf=pf, log_mass=float(log_mass))

    def _hazard(self, rl: int) -> float:
        r = torch.tensor([rl], dtype=torch.float64)
        val = self.config.hazard(r)[0].item()
        return float(max(min(val, 1.0 - 1e-12), 1e-12))

    def _expert_theta_mean(self, e: ExpertPaperPF) -> float:
        return float(np.mean(e.pf.theta))

    def _closest_by_run_length(self, target_rl: int) -> Optional[ExpertPaperPF]:
        if not self.experts:
            return None
        return min(self.experts, key=lambda e: abs(e.run_length - target_rl))

    def _prune_keep_anchor(self, anchor_run_length: int, max_experts: int) -> None:
        if len(self.experts) <= max_experts:
            return
        anchor_idx: Optional[int] = None
        best_diff = 10 ** 9
        for i, e in enumerate(self.experts):
            diff = abs(e.run_length - anchor_run_length)
            if diff < best_diff:
                best_diff = diff
                anchor_idx = i
        sorted_idx = sorted(range(len(self.experts)), key=lambda i: self.experts[i].log_mass, reverse=True)
        kept: List[int] = []
        for idx in sorted_idx:
            if idx == anchor_idx or len(kept) < max_experts - 1:
                kept.append(idx)
            if len(kept) >= max_experts:
                break
        if anchor_idx is not None and anchor_idx not in kept:
            kept[-1] = anchor_idx
        kept = sorted(set(kept))
        self.experts = [self.experts[i] for i in kept]

    def _theta_test_pass(self, anchor_e: Optional[ExpertPaperPF], cand_e: Optional[ExpertPaperPF]) -> Tuple[bool, Optional[float]]:
        if self.restart_criteria != "theta_test" or anchor_e is None or cand_e is None:
            return True, 0.0
        a = np.asarray(anchor_e.pf.theta, dtype=np.float64).reshape(-1)
        b = np.asarray(cand_e.pf.theta, dtype=np.float64).reshape(-1)
        if a.size == 0 or b.size == 0:
            return False, float("nan")
        if str(getattr(self.config, "restart_theta_test", "energy")) == "credible":
            z = float(getattr(self.config, "restart_cred_z", 2.0))
            frac = float(getattr(self.config, "restart_cred_frac", 0.5))
            mu_a, sd_a = float(a.mean()), float(a.std(ddof=0))
            mu_b, sd_b = float(b.mean()), float(b.std(ddof=0))
            lo_a, hi_a = mu_a - z * sd_a, mu_a + z * sd_a
            lo_b, hi_b = mu_b - z * sd_b, mu_b + z * sd_b
            nonoverlap = float((hi_a < lo_b) or (hi_b < lo_a))
            return nonoverlap >= frac, nonoverlap
        energy = float(abs(a.mean() - b.mean()))
        tau = float(getattr(self.config, "restart_theta_tau", 0.1))
        return energy > tau, energy

    def ump_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> Tuple[List[float], Dict[int, np.ndarray]]:
        out: List[float] = []
        cache: Dict[int, np.ndarray] = {}
        for e in self.experts:
            loglik_particles = e.pf.compute_loglik_batch(X_batch_np, Y_batch_np)
            logmix = _logsumexp_np(loglik_particles) - math.log(max(len(loglik_particles), 1))
            out.append(float(logmix))
            cache[id(e)] = loglik_particles
        return out, cache

    def predict_batch(self, X_batch_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.experts:
            n = int(np.asarray(X_batch_np).shape[0])
            return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
        masses = np.asarray([math.exp(e.log_mass) for e in self.experts], dtype=np.float64)
        masses = masses / np.clip(masses.sum(), 1e-300, None)
        mu_acc = None
        second_acc = None
        for w_e, e in zip(masses, self.experts):
            mu_e, var_e = e.pf.predict_batch(X_batch_np)
            if mu_acc is None:
                mu_acc = w_e * mu_e
                second_acc = w_e * (var_e + mu_e * mu_e)
            else:
                mu_acc = mu_acc + w_e * mu_e
                second_acc = second_acc + w_e * (var_e + mu_e * mu_e)
        var_acc = np.clip(second_acc - mu_acc * mu_acc, 1e-12, None)
        return mu_acc, var_acc

    def update_batch(self, X_batch_np: np.ndarray, Y_batch_np: np.ndarray) -> Dict[str, Any]:
        X_batch_np = np.asarray(X_batch_np, dtype=np.float64)
        Y_batch_np = np.asarray(Y_batch_np, dtype=np.float64).reshape(-1)
        batch_size = int(Y_batch_np.shape[0])

        if len(self.experts) == 0:
            self.experts.append(self._spawn_new_expert(log_mass=0.0))
            self.restart_start_time = 0
            self.t = 0
            self._last_restart_t = -(10 ** 9)

        log_umps, loglik_cache = self.ump_batch(X_batch_np, Y_batch_np)
        experts_pre = list(self.experts)
        log_umps_pre = list(log_umps)
        idx_pre = {id(e): i for i, e in enumerate(experts_pre)}

        def get_pre_log_ump(e: Optional[ExpertPaperPF]):
            if e is None:
                return None
            j = idx_pre.get(id(e), None)
            return None if j is None else float(log_umps_pre[j])

        prev_log_mass = np.asarray([e.log_mass for e in self.experts], dtype=np.float64)
        hazards = np.asarray([self._hazard(e.run_length) for e in self.experts], dtype=np.float64)
        growth = prev_log_mass + np.log(np.clip(1.0 - hazards, 1e-12, None)) + np.asarray(log_umps, dtype=np.float64)
        cp = _logsumexp_np(prev_log_mass + np.log(np.clip(hazards, 1e-12, None)) + np.asarray(log_umps, dtype=np.float64))

        for i, e in enumerate(self.experts):
            e.run_length += batch_size
            e.log_mass = float(growth[i])

        self.experts.append(self._spawn_new_expert(log_mass=float(cp)))
        masses = np.asarray([e.log_mass for e in self.experts], dtype=np.float64)
        log_Z = _logsumexp_np(masses)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)
        anchor_e = self._closest_by_run_length(anchor_rl)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        best_other_mass = 0.0
        s_star: Optional[int] = None
        cand_e: Optional[ExpertPaperPF] = None
        for e in self.experts:
            s = t_now - e.run_length
            if s > r_old:
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = int(s)
                    cand_e = e

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

        did_restart = False
        msg_mode = "NO_RESTART"
        theta_pass, theta_stat = self._theta_test_pass(anchor_e, cand_e)

        if (
            getattr(self.config, "use_restart", True)
            and s_star is not None
            and theta_pass
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        ):
            did_restart = True
            self._last_restart_t = t_now
            if getattr(self.config, "use_backdated_restart", False):
                self.restart_start_time = int(s_star)
                new_anchor_rl = max(t_now - self.restart_start_time, 0)
                keep_e = next((e for e in self.experts if e.run_length == new_anchor_rl), None)
                if keep_e is None and self.experts:
                    keep_e = min(self.experts, key=lambda e: abs(e.run_length - new_anchor_rl))
                    keep_e.run_length = new_anchor_rl
                self.experts = [keep_e] if keep_e is not None else self.experts[:1]
                msg_mode = "BACKDATED"
            else:
                self.restart_start_time = t_now
                self.experts = [self._spawn_new_expert(log_mass=0.0)]
                msg_mode = "ALGO2"
            if self.on_restart is not None and self.notify_on_restart:
                self.on_restart(
                    int(t_now),
                    int(self.restart_start_time),
                    int(s_star) if s_star is not None else None,
                    int(anchor_rl),
                    float(p_anchor),
                    float(best_other_mass),
                )

        if not did_restart:
            anchor_run_length = max(t_now - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        pf_diags: List[Dict[str, Any]] = []
        for e in self.experts:
            info = e.pf.step_batch(X_batch_np, Y_batch_np)
            pf_diags.append(
                {
                    "ess": float(e.pf.cfg.num_particles),
                    "gini": 0.0,
                    "resampled": True,
                    "theta_mean_post": float(np.mean(info["theta_particles"])),
                    "l_mean_post": float(np.mean(info["lengthscale_particles"])),
                }
            )

        self.t += batch_size
        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)
        if log_umps:
            self.prev_max_ump = max(log_umps)

        experts_debug: List[Dict[str, Any]] = []
        for idx_e, e in enumerate(self.experts):
            mass = math.exp(e.log_mass)
            lu = float(log_umps[idx_e]) if idx_e < len(log_umps) else None
            experts_debug.append(
                {
                    "index": idx_e,
                    "run_length": int(e.run_length),
                    "start_time": int(t_now - e.run_length),
                    "log_mass": float(e.log_mass),
                    "mass": float(mass),
                    "theta_mean": [float(np.mean(e.pf.theta))],
                    "log_ump": lu,
                }
            )

        log_ump_anchor = get_pre_log_ump(anchor_e)
        log_ump_cand = get_pre_log_ump(cand_e)
        delta_ll_pair = None if (log_ump_anchor is None or log_ump_cand is None) else float(log_ump_cand - log_ump_anchor)
        log_odds_mass = None if (anchor_e is None or cand_e is None) else float(cand_e.log_mass - anchor_e.log_mass)
        h_log = float(math.log1p(self.restart_margin))

        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": pf_diags,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
            "experts_debug": experts_debug,
            "anchor_rl": int(anchor_e.run_length) if anchor_e is not None else None,
            "cand_rl": int(cand_e.run_length) if cand_e is not None else None,
            "delta_ll_pair": delta_ll_pair,
            "log_odds_mass": log_odds_mass,
            "h_log": h_log,
            "log_ump_anchor": log_ump_anchor,
            "log_ump_cand": log_ump_cand,
            "theta_stat": theta_stat,
            "restart_mode": msg_mode,
        }

    def aggregate_particles(self, quantile: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.experts:
            nan = np.asarray([float("nan")], dtype=np.float64)
            return nan.copy(), nan.copy(), nan.copy(), nan.copy()
        theta_parts = []
        weight_parts = []
        for e in self.experts:
            theta = np.asarray(e.pf.theta, dtype=np.float64).reshape(-1)
            w = np.full(theta.shape[0], math.exp(e.log_mass) / max(theta.shape[0], 1), dtype=np.float64)
            theta_parts.append(theta)
            weight_parts.append(w)
        theta_all = np.concatenate(theta_parts)
        weight_all = np.concatenate(weight_parts)
        weight_all = weight_all / np.clip(weight_all.sum(), 1e-300, None)
        mean = float(np.sum(weight_all * theta_all))
        var = float(np.sum(weight_all * (theta_all - mean) ** 2))
        idx = np.argsort(theta_all)
        xs = theta_all[idx]
        ws = weight_all[idx]
        cw = np.cumsum(ws)
        alpha = (1.0 - quantile) / 2.0
        lo = float(xs[np.searchsorted(cw, alpha, side="left")])
        hi = float(xs[np.searchsorted(cw, 1.0 - alpha, side="left")])
        return (
            np.asarray([mean], dtype=np.float64),
            np.asarray([var], dtype=np.float64),
            np.asarray([lo], dtype=np.float64),
            np.asarray([hi], dtype=np.float64),
        )
