from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import torch
import gpytorch

from .configs import ModelConfig, PFConfig
from .delta_gp import GPyTorchDeltaState, fit_gpytorch_delta, fit_scale_rbf_gpytorch_hyper
from .emulator import Emulator
from .particle_specific_discrepancy import (
    KernelHyperSpec,
    ParticleSpecificBasisDeltaState,
    ParticleSpecificBatchSupportOnlineBPCDeltaState,
    ParticleSpecificExactOnlineBPCDeltaState,
    ParticleSpecificFixedSupportOnlineBPCDeltaState,
    ParticleSpecificDynamicBasisDeltaState,
    ParticleSpecificGPDeltaState,
    ParticleSpecificMCInducingDeltaState,
    ParticleSpecificOnlineGPDeltaState,
)
from .particles import ParticleSet
from .restart_bocpd_hybrid_260319_gpytorch import BOCPD as HybridBOCPD
from .restart_bocpd_debug_260115_gpytorch import Expert


class BOCPD(HybridBOCPD):
    """
    Opt-in hybrid BOCPD variant with discrepancy-memory refresh policies.

    The refresh patch is intentionally isolated from BOCPD/PF logic:
    - `super().update_batch(...)` runs the original hybrid implementation.
    - only when BOCPD did not restart do we optionally refresh the anchor
      expert's discrepancy history.
    - any refresh-policy failure degrades to a no-op so existing runs stay intact.

    This file also supports richer BOCPD-side discrepancy states:
    - shared expert GP (current behavior)
    - particle-specific GP with shared fitted hyperparameters
    - particle-specific GP with a small shared hyperparameter pool
    - particle-specific Bayesian basis discrepancy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cusum = bool(getattr(self.config, "use_cusum", True))
        self.cusum_threshold = float(getattr(self.config, "cusum_threshold", 10.0))
        self.cusum_recent_obs = max(1, int(getattr(self.config, "cusum_recent_obs", 20)))
        self.cusum_cov_eps = max(float(getattr(self.config, "cusum_cov_eps", 1e-6)), 1e-12)

        mode = str(getattr(self.config, "cusum_mode", "cumulative")).lower()
        if mode in {"gate", "single_step_gate", "standardized_gate", "single_step_standardized_gate"}:
            mode = "standardized_gate"
        else:
            mode = "cumulative"
        self.cusum_mode = mode
        self.standardized_gate_threshold = float(getattr(self.config, "standardized_gate_threshold", 3.0))
        self.standardized_gate_consecutive = max(1, int(getattr(self.config, "standardized_gate_consecutive", 1)))

        self._cusum_stat = 0.0
        self._cusum_tau = 0
        self._gate_hits = 0
        self._cusum_prev_anchor_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _particle_delta_mode(self) -> str:
        mode = str(getattr(self.config, "particle_delta_mode", "shared_gp")).lower()
        aliases = {
            "shared": "shared_gp",
            "shared_gp": "shared_gp",
            "particle_gp": "particle_gp_shared_hyper",
            "particle_gp_shared": "particle_gp_shared_hyper",
            "particle_gp_shared_hyper": "particle_gp_shared_hyper",
            "particle_gp_online": "particle_gp_online_shared_hyper",
            "particle_gp_online_shared": "particle_gp_online_shared_hyper",
            "particle_gp_online_shared_hyper": "particle_gp_online_shared_hyper",
            "particle_gp_dynamic": "particle_gp_dynamic_shared_hyper",
            "particle_gp_dynamic_shared": "particle_gp_dynamic_shared_hyper",
            "particle_gp_dynamic_shared_hyper": "particle_gp_dynamic_shared_hyper",
            "particle_gp_bpc": "particle_gp_online_bpc_shared_hyper",
            "particle_gp_online_bpc": "particle_gp_online_bpc_shared_hyper",
            "particle_gp_online_bpc_shared": "particle_gp_online_bpc_shared_hyper",
            "particle_gp_online_bpc_shared_hyper": "particle_gp_online_bpc_shared_hyper",
            "particle_gp_bpc_exact": "particle_gp_online_bpc_exact_shared_hyper",
            "particle_gp_online_bpc_exact": "particle_gp_online_bpc_exact_shared_hyper",
            "particle_gp_online_bpc_exact_shared": "particle_gp_online_bpc_exact_shared_hyper",
            "particle_gp_online_bpc_exact_shared_hyper": "particle_gp_online_bpc_exact_shared_hyper",
            "particle_gp_fixedsupport_online_bpc": "particle_gp_fixedsupport_online_bpc_shared_hyper",
            "particle_gp_fixedsupport_online_bpc_shared": "particle_gp_fixedsupport_online_bpc_shared_hyper",
            "particle_gp_fixedsupport_online_bpc_shared_hyper": "particle_gp_fixedsupport_online_bpc_shared_hyper",
            "particle_mc_inducing": "particle_mc_inducing",
            "particle_mc": "particle_mc_inducing",
            "particle_mc_shared_hyper": "particle_mc_inducing",
            "particle_gp_hyper_pool": "particle_gp_hyper_pool",
            "particle_basis": "particle_basis",
            "basis": "particle_basis",
        }
        return aliases.get(mode, "shared_gp")

    def _anchor_expert_after_update(self) -> Optional[Expert]:
        if len(self.experts) == 0:
            return None
        anchor_run_length = max(int(self.t) - int(self.restart_start_time), 0)
        return self._closest_by_run_length(anchor_run_length)

    def _posterior_summary(self, e: Expert) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        ps: ParticleSet = e.pf.particles
        theta = ps.theta
        if theta.numel() == 0:
            return None

        w = ps.weights().reshape(-1)
        if w.numel() != theta.shape[0]:
            return None
        if torch.isnan(w).any() or torch.isinf(w).any():
            return None

        w_sum = torch.clamp(w.sum(), min=1e-12)
        w = w / w_sum
        mean = (w.view(-1, 1) * theta).sum(dim=0)
        centered = theta - mean
        cov = centered.transpose(0, 1) @ (w.view(-1, 1) * centered)
        return mean.detach().clone(), cov.detach().clone()

    def _cusum_distance(
        self,
        prev_stats: Tuple[torch.Tensor, torch.Tensor],
        cur_stats: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        prev_mean, prev_cov = prev_stats
        cur_mean, _ = cur_stats
        delta = (cur_mean - prev_mean).reshape(-1, 1)
        dim = int(delta.shape[0])
        eye = torch.eye(dim, dtype=prev_cov.dtype, device=prev_cov.device)
        reg_cov = prev_cov + self.cusum_cov_eps * eye
        try:
            solved = torch.linalg.solve(reg_cov, delta)
        except Exception:
            solved = torch.linalg.pinv(reg_cov) @ delta
        score = float((delta.transpose(0, 1) @ solved).reshape(()).item())
        if not math.isfinite(score):
            raise ValueError("non-finite refresh score")
        return max(score, 0.0)

    def _refresh_anchor_discrepancy(self, anchor_e: Expert, model_cfg: ModelConfig) -> bool:
        if anchor_e.X_hist.numel() == 0 or anchor_e.y_hist.numel() == 0:
            return False

        old_X = anchor_e.X_hist
        old_y = anchor_e.y_hist
        old_delta = anchor_e.delta_state

        keep_n = min(int(anchor_e.X_hist.shape[0]), self.cusum_recent_obs)
        recent_X = anchor_e.X_hist[-keep_n:].clone()
        recent_y = anchor_e.y_hist[-keep_n:].clone()

        try:
            anchor_e.X_hist = recent_X
            anchor_e.y_hist = recent_y
            delta_mode = self._delta_update_mode(model_cfg)
            if delta_mode in {"online", "online_dynamic", "online_inducing"}:
                particle_mode = self._particle_delta_mode()
                if delta_mode == "online" and particle_mode == "shared_gp":
                    ok = self._truncate_shared_online_delta(anchor_e, keep_n, model_cfg)
                elif delta_mode == "online_dynamic" and particle_mode == "shared_gp":
                    ok = self._truncate_shared_dynamic_delta(anchor_e, keep_n, model_cfg)
                elif delta_mode == "online_inducing" and particle_mode == "shared_gp":
                    ok = self._truncate_shared_inducing_delta(anchor_e, keep_n, model_cfg)
                elif delta_mode == "online" and particle_mode == "particle_gp_online_shared_hyper":
                    ok = self._truncate_particle_online_delta(anchor_e, keep_n, model_cfg)
                elif delta_mode == "online_dynamic" and particle_mode == "particle_gp_dynamic_shared_hyper":
                    ok = self._truncate_particle_dynamic_delta(anchor_e, keep_n, model_cfg)
                else:
                    ok = False
                if ok:
                    return True
            self._reset_delta_for_expert(anchor_e, model_cfg)
            if keep_n >= 3 and anchor_e.delta_state is None and old_delta is not None:
                raise RuntimeError("delta refit returned None on recent memory")
            return True
        except Exception:
            anchor_e.X_hist = old_X
            anchor_e.y_hist = old_y
            anchor_e.delta_state = old_delta
            return False

    def _shared_residual_history(self, e: Expert, model_cfg: ModelConfig):
        if e.X_hist.numel() == 0:
            return None
        X_hist = e.X_hist
        Y_hist = e.y_hist
        mu_eta_all, _ = self._last_emulator.predict(X_hist, e.pf.particles.theta)
        weights = e.pf.particles.weights().view(1, -1)
        if torch.isnan(weights).any() or weights.sum() < 1e-30:
            return None
        if mu_eta_all.dim() == 2:
            eta_mix_all = (weights * mu_eta_all).sum(dim=1)
        else:
            eta_mix_all = (weights.unsqueeze(-1) * mu_eta_all).sum(dim=1)
        resid_all = Y_hist - model_cfg.rho * eta_mix_all
        resid_for_delta = (
            resid_all.mean(dim=-1)
            if resid_all.dim() > 1 and resid_all.shape[-1] > 1
            else resid_all.reshape(-1)
        )
        if torch.isnan(resid_for_delta).any() or torch.isinf(resid_for_delta).any():
            return None
        return X_hist, Y_hist, resid_for_delta

    def _fit_shared_delta_hyper(self, X_hist: torch.Tensor, resid: torch.Tensor, model_cfg: ModelConfig) -> KernelHyperSpec:
        dx = int(X_hist.shape[1])
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dx)
        scale_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        try:
            ls0 = torch.as_tensor(model_cfg.delta_kernel.lengthscale, device=X_hist.device, dtype=X_hist.dtype).reshape(-1)
            if ls0.numel() == 1:
                ls0 = ls0.repeat(dx)
            if ls0.numel() == dx:
                scale_kernel.base_kernel.lengthscale = ls0.reshape(1, -1)
        except Exception:
            pass
        try:
            scale_kernel.outputscale = float(getattr(model_cfg.delta_kernel, "variance", 0.1))
        except Exception:
            pass
        state = GPyTorchDeltaState(
            X=X_hist,
            y=resid,
            kernel=scale_kernel,
            noise=max(float(model_cfg.delta_kernel.noise), 1e-6),
        )
        fitted = fit_gpytorch_delta(state, max_iter=80)
        lengthscale = fitted.model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        variance = float(fitted.model.covar_module.outputscale.detach().reshape(()).item())
        noise_attr = getattr(fitted.likelihood, "noise", None)
        if noise_attr is None:
            noise = max(float(model_cfg.delta_kernel.noise), 1e-6)
        else:
            noise = float(torch.as_tensor(noise_attr).detach().reshape(-1).mean().item())
        return KernelHyperSpec(lengthscale=lengthscale, variance=variance, noise=noise)

    def _build_hyper_pool(self, shared_spec: KernelHyperSpec) -> List[KernelHyperSpec]:
        raw_pool = getattr(self.config, "particle_gp_hyper_candidates", None)
        if raw_pool:
            pool: List[KernelHyperSpec] = []
            for item in raw_pool:
                if isinstance(item, dict):
                    pool.append(
                        KernelHyperSpec(
                            lengthscale=torch.as_tensor(item.get("lengthscale", shared_spec.lengthscale), device=shared_spec.lengthscale.device, dtype=shared_spec.lengthscale.dtype),
                            variance=float(item.get("variance", shared_spec.variance)),
                            noise=float(item.get("noise", shared_spec.noise)),
                        )
                    )
            if pool:
                return pool
        ls_scales = [0.5, 1.0, 2.0]
        return [
            KernelHyperSpec(
                lengthscale=shared_spec.lengthscale * scale,
                variance=shared_spec.variance,
                noise=shared_spec.noise,
            )
            for scale in ls_scales
        ]

    def _update_particle_online_delta(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificOnlineGPDeltaState):
            e.delta_state = ParticleSpecificOnlineGPDeltaState(
                theta_particles=e.pf.particles.theta.detach().clone(),
                emulator=emulator,
                rho=model_cfg.rho,
                min_points=self._delta_online_min_points(model_cfg),
            )
        state: ParticleSpecificOnlineGPDeltaState = e.delta_state
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )
        shared_resid = self._shared_batch_residual(e, X_batch, Y_batch, emulator, model_cfg)
        if shared_resid is None:
            return
        buf_X, buf_y = self._append_delta_online_buffer(e, X_batch, shared_resid)
        if state.hyper_specs is None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = e.delta_meta.get("delta_online_hyper_spec")
            if spec is None:
                spec = fit_scale_rbf_gpytorch_hyper(
                    buf_X,
                    buf_y,
                    noise=float(model_cfg.delta_kernel.noise),
                    lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                    variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                    max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
                )
                e.delta_meta["delta_online_hyper_spec"] = spec
            state.set_hyper_specs([
                KernelHyperSpec(
                    lengthscale=spec.lengthscale,
                    variance=spec.variance,
                    noise=spec.noise,
                )
            ])

    def _truncate_particle_online_delta(self, e: Expert, keep_n: int, model_cfg: ModelConfig) -> bool:
        if not isinstance(e.delta_state, ParticleSpecificOnlineGPDeltaState):
            return False
        buf_X, buf_y = self._trim_delta_online_buffer(e, keep_n)
        e.delta_state.truncate_recent(keep_n)
        if e.delta_state.hyper_specs is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = fit_scale_rbf_gpytorch_hyper(
                buf_X,
                buf_y,
                noise=float(model_cfg.delta_kernel.noise),
                lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
            )
            e.delta_meta["delta_online_hyper_spec"] = spec
            e.delta_state.set_hyper_specs([
                KernelHyperSpec(
                    lengthscale=spec.lengthscale,
                    variance=spec.variance,
                    noise=spec.noise,
                )
            ])
        return True

    def _update_particle_dynamic_delta(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificDynamicBasisDeltaState):
            e.delta_state = ParticleSpecificDynamicBasisDeltaState(
                theta_particles=e.pf.particles.theta.detach().clone(),
                emulator=emulator,
                rho=model_cfg.rho,
                min_points=self._delta_online_min_points(model_cfg),
                num_features=self._delta_dynamic_num_features(model_cfg),
                forgetting=self._delta_dynamic_forgetting(model_cfg),
                process_noise_scale=self._delta_dynamic_process_noise_scale(model_cfg),
                prior_var_scale=self._delta_dynamic_prior_var_scale(model_cfg),
                buffer_max_points=self._delta_dynamic_buffer_max_points(model_cfg),
            )
        state: ParticleSpecificDynamicBasisDeltaState = e.delta_state
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )
        shared_resid = self._shared_batch_residual(e, X_batch, Y_batch, emulator, model_cfg)
        if shared_resid is None:
            return
        buf_X, buf_y = self._append_delta_online_buffer(e, X_batch, shared_resid)
        max_buf = self._delta_dynamic_buffer_max_points(model_cfg)
        if buf_X.shape[0] > max_buf:
            buf_X, buf_y = self._trim_delta_online_buffer(e, max_buf)
        if state.hyper_spec is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = e.delta_meta.get("delta_online_hyper_spec")
            if spec is None:
                spec = fit_scale_rbf_gpytorch_hyper(
                    buf_X,
                    buf_y,
                    noise=float(model_cfg.delta_kernel.noise),
                    lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                    variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                    max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
                )
                e.delta_meta["delta_online_hyper_spec"] = spec
            state.set_hyper_spec(
                KernelHyperSpec(
                    lengthscale=spec.lengthscale,
                    variance=spec.variance,
                    noise=spec.noise,
                )
            )

    def _update_particle_bpc_delta(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificBatchSupportOnlineBPCDeltaState):
            e.delta_state = ParticleSpecificBatchSupportOnlineBPCDeltaState(
                theta_particles=e.pf.particles.theta.detach().clone(),
                emulator=emulator,
                rho=model_cfg.rho,
                min_points=self._delta_online_min_points(model_cfg),
                lambda_delta=self._delta_bpc_lambda(model_cfg),
            )
        state: ParticleSpecificBatchSupportOnlineBPCDeltaState = e.delta_state
        shared_resid = self._shared_batch_residual(e, X_batch, Y_batch, emulator, model_cfg)
        if shared_resid is None:
            return
        buf_X, buf_y = self._append_delta_online_buffer(e, X_batch, shared_resid)
        max_buf = self._delta_dynamic_buffer_max_points(model_cfg)
        if buf_X is not None and buf_X.shape[0] > max_buf:
            buf_X, buf_y = self._trim_delta_online_buffer(e, max_buf)
        if state.hyper_spec is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = e.delta_meta.get("delta_online_hyper_spec")
            if spec is None:
                spec = fit_scale_rbf_gpytorch_hyper(
                    buf_X,
                    buf_y,
                    noise=float(model_cfg.delta_kernel.noise),
                    lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                    variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                    max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
                )
                e.delta_meta["delta_online_hyper_spec"] = spec
            state.set_hyper_spec(KernelHyperSpec(lengthscale=spec.lengthscale, variance=spec.variance, noise=spec.noise))
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )

    def _update_particle_bpc_exact_delta(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificExactOnlineBPCDeltaState):
            e.delta_state = ParticleSpecificExactOnlineBPCDeltaState(
                theta_particles=e.pf.particles.theta.detach().clone(),
                emulator=emulator,
                rho=model_cfg.rho,
                min_points=self._delta_online_min_points(model_cfg),
                lambda_delta=self._delta_bpc_lambda(model_cfg),
            )
        state: ParticleSpecificExactOnlineBPCDeltaState = e.delta_state
        shared_resid = self._shared_batch_residual(e, X_batch, Y_batch, emulator, model_cfg)
        if shared_resid is None:
            return
        buf_X, buf_y = self._append_delta_online_buffer(e, X_batch, shared_resid)
        if state.hyper_spec is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = e.delta_meta.get("delta_online_hyper_spec")
            if spec is None:
                spec = fit_scale_rbf_gpytorch_hyper(
                    buf_X,
                    buf_y,
                    noise=float(model_cfg.delta_kernel.noise),
                    lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                    variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                    max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
                )
                e.delta_meta["delta_online_hyper_spec"] = spec
            state.set_hyper_spec(KernelHyperSpec(lengthscale=spec.lengthscale, variance=spec.variance, noise=spec.noise))
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )

    def _update_particle_bpc_fixedsupport_delta(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificFixedSupportOnlineBPCDeltaState):
            e.delta_state = ParticleSpecificFixedSupportOnlineBPCDeltaState(
                theta_particles=e.pf.particles.theta.detach().clone(),
                emulator=emulator,
                rho=model_cfg.rho,
                min_points=self._delta_online_min_points(model_cfg),
                num_support=self._delta_inducing_num_points(model_cfg),
                lambda_delta=self._delta_bpc_lambda(model_cfg),
                obs_noise_var=self._delta_bpc_obs_noise_var(model_cfg),
                add_kernel_noise_to_predict=self._delta_bpc_predict_add_kernel_noise(model_cfg),
            )
        state: ParticleSpecificFixedSupportOnlineBPCDeltaState = e.delta_state
        shared_resid = self._shared_batch_residual(e, X_batch, Y_batch, emulator, model_cfg)
        if shared_resid is None:
            return
        buf_X, buf_y = self._append_delta_online_buffer(e, X_batch, shared_resid)
        if state.hyper_spec is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = e.delta_meta.get("delta_online_hyper_spec")
            if spec is None:
                spec = fit_scale_rbf_gpytorch_hyper(
                    buf_X,
                    buf_y,
                    noise=float(model_cfg.delta_kernel.noise),
                    lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                    variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                    max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
                )
                e.delta_meta["delta_online_hyper_spec"] = spec
            state.set_hyper_spec(KernelHyperSpec(lengthscale=spec.lengthscale, variance=spec.variance, noise=spec.noise))
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )

    def _update_particle_mc_inducing(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        if not isinstance(e.delta_state, ParticleSpecificMCInducingDeltaState):
            self._reset_delta_for_expert(e, model_cfg)
            if not isinstance(e.delta_state, ParticleSpecificMCInducingDeltaState):
                return
        state: ParticleSpecificMCInducingDeltaState = e.delta_state
        state.append_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            theta_particles=e.pf.particles.theta.detach().clone(),
            ancestor_indices=pf_diag.get("ancestor_indices"),
            emulator=emulator,
            rho=model_cfg.rho,
        )
        e.delta_meta["mc_batches_since_refresh"] = int(e.delta_meta.get("mc_batches_since_refresh", 0)) + 1
        refresh_every = self._delta_mc_refresh_every(model_cfg)
        if self._delta_update_mode(model_cfg) == "online_particle_mc_inducing_refresh" and refresh_every > 0 and e.delta_meta["mc_batches_since_refresh"] >= refresh_every:
            self._reset_delta_for_expert(e, model_cfg)

    def _truncate_particle_dynamic_delta(self, e: Expert, keep_n: int, model_cfg: ModelConfig) -> bool:
        if not isinstance(e.delta_state, ParticleSpecificDynamicBasisDeltaState):
            return False
        buf_X, buf_y = self._trim_delta_online_buffer(e, keep_n)
        e.delta_state.truncate_recent(keep_n)
        if e.delta_state.hyper_spec is None and buf_X is not None and buf_y is not None and buf_X.shape[0] >= self._delta_online_min_points(model_cfg):
            spec = fit_scale_rbf_gpytorch_hyper(
                buf_X,
                buf_y,
                noise=float(model_cfg.delta_kernel.noise),
                lengthscale_init=getattr(model_cfg.delta_kernel, "lengthscale", 1.0),
                variance_init=float(getattr(model_cfg.delta_kernel, "variance", 0.1)),
                max_iter=int(getattr(model_cfg, "delta_online_init_max_iter", 80)),
            )
            e.delta_meta["delta_online_hyper_spec"] = spec
            e.delta_state.set_hyper_spec(
                KernelHyperSpec(
                    lengthscale=spec.lengthscale,
                    variance=spec.variance,
                    noise=spec.noise,
                )
            )
        return True

    def _update_delta_after_batch(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_diag: Dict[str, Any],
    ) -> None:
        mode = self._particle_delta_mode()
        if mode == "shared_gp":
            return super()._update_delta_after_batch(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
        if not getattr(model_cfg, "refit_delta_every_batch", True):
            return
        delta_mode = self._delta_update_mode(model_cfg)
        if delta_mode == "online" and mode == "particle_gp_online_shared_hyper":
            self._update_particle_online_delta(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        if delta_mode == "online_dynamic" and mode == "particle_gp_dynamic_shared_hyper":
            self._update_particle_dynamic_delta(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        if delta_mode == "online_bpc" and mode == "particle_gp_online_bpc_shared_hyper":
            self._update_particle_bpc_delta(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        if delta_mode == "online_bpc_exact" and mode == "particle_gp_online_bpc_exact_shared_hyper":
            self._update_particle_bpc_exact_delta(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        if delta_mode == "online_bpc_fixedsupport_exact" and mode == "particle_gp_fixedsupport_online_bpc_shared_hyper":
            self._update_particle_bpc_fixedsupport_delta(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        if delta_mode in {"online_particle_mc_inducing", "online_particle_mc_inducing_refresh"} and mode == "particle_mc_inducing":
            self._update_particle_mc_inducing(e, X_batch, Y_batch, emulator, model_cfg, pf_diag)
            return
        self._reset_delta_for_expert(e, model_cfg)

    def _reset_delta_for_expert(self, e: Expert, model_cfg: ModelConfig) -> None:
        mode = self._particle_delta_mode()
        if mode == "shared_gp":
            return super()._reset_delta_for_expert(e, model_cfg)

        hist_info = self._shared_residual_history(e, model_cfg)
        if hist_info is None:
            e.delta_state = None
            return
        X_hist, Y_hist, resid_shared = hist_info
        theta_particles = e.pf.particles.theta.detach().clone()

        try:
            if mode == "particle_gp_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                e.delta_state = ParticleSpecificGPDeltaState(
                    X_hist=X_hist,
                    Y_hist=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_specs=[shared_spec],
                )
                return

            if mode == "particle_gp_online_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                e.delta_state = ParticleSpecificOnlineGPDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_specs=[shared_spec],
                    min_points=self._delta_online_min_points(model_cfg),
                )
                e.delta_state.append_batch(
                    X_batch=X_hist,
                    Y_batch=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                return

            if mode == "particle_gp_dynamic_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                dynamic_state = ParticleSpecificDynamicBasisDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_spec=shared_spec,
                    min_points=self._delta_online_min_points(model_cfg),
                    num_features=self._delta_dynamic_num_features(model_cfg),
                    forgetting=self._delta_dynamic_forgetting(model_cfg),
                    process_noise_scale=self._delta_dynamic_process_noise_scale(model_cfg),
                    prior_var_scale=self._delta_dynamic_prior_var_scale(model_cfg),
                    buffer_max_points=self._delta_dynamic_buffer_max_points(model_cfg),
                )
                dynamic_state.append_batch(
                    X_batch=X_hist,
                    Y_batch=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                e.delta_state = dynamic_state
                return

            if mode == "particle_gp_online_bpc_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                bpc_state = ParticleSpecificBatchSupportOnlineBPCDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_spec=KernelHyperSpec(
                        lengthscale=shared_spec.lengthscale,
                        variance=shared_spec.variance,
                        noise=shared_spec.noise,
                    ),
                    min_points=self._delta_online_min_points(model_cfg),
                    lambda_delta=self._delta_bpc_lambda(model_cfg),
                )
                batch_sizes = e.delta_meta.get("delta_online_batch_sizes", [])
                last_size = int(batch_sizes[-1]) if batch_sizes else int(X_hist.shape[0])
                last_size = max(1, min(last_size, int(X_hist.shape[0])))
                bpc_state.append_batch(
                    X_batch=X_hist[-last_size:],
                    Y_batch=Y_hist[-last_size:],
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                e.delta_state = bpc_state
                e.delta_meta["delta_online_hyper_spec"] = shared_spec
                return

            if mode == "particle_gp_online_bpc_exact_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                bpc_state = ParticleSpecificExactOnlineBPCDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_spec=KernelHyperSpec(
                        lengthscale=shared_spec.lengthscale,
                        variance=shared_spec.variance,
                        noise=shared_spec.noise,
                    ),
                    min_points=self._delta_online_min_points(model_cfg),
                    lambda_delta=self._delta_bpc_lambda(model_cfg),
                )
                bpc_state.append_batch(
                    X_batch=X_hist,
                    Y_batch=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                e.delta_state = bpc_state
                e.delta_meta["delta_online_hyper_spec"] = shared_spec
                return
            if mode == "particle_gp_fixedsupport_online_bpc_shared_hyper":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                bpc_state = ParticleSpecificFixedSupportOnlineBPCDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_spec=KernelHyperSpec(
                        lengthscale=shared_spec.lengthscale,
                        variance=shared_spec.variance,
                        noise=shared_spec.noise,
                    ),
                    min_points=self._delta_online_min_points(model_cfg),
                    num_support=self._delta_inducing_num_points(model_cfg),
                    lambda_delta=self._delta_bpc_lambda(model_cfg),
                    obs_noise_var=self._delta_bpc_obs_noise_var(model_cfg),
                    add_kernel_noise_to_predict=self._delta_bpc_predict_add_kernel_noise(model_cfg),
                )
                bpc_state.append_batch(
                    X_batch=X_hist,
                    Y_batch=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                e.delta_state = bpc_state
                e.delta_meta["delta_online_hyper_spec"] = shared_spec
                return

            if mode == "particle_mc_inducing":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                mc_state = ParticleSpecificMCInducingDeltaState(
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_spec=shared_spec,
                    min_points=self._delta_online_min_points(model_cfg),
                    num_inducing=self._delta_mc_num_inducing_points(model_cfg),
                    num_mc_particles=self._delta_mc_num_particles(model_cfg),
                    resample_ess_ratio=self._delta_mc_resample_ess_ratio(model_cfg),
                    include_conditional_var=self._delta_mc_include_conditional_var(model_cfg),
                )
                mc_state.refresh_from_history(
                    X_hist=X_hist,
                    Y_hist=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                )
                e.delta_state = mc_state
                e.delta_meta["delta_online_hyper_spec"] = shared_spec
                e.delta_meta["mc_batches_since_refresh"] = 0
                return

            if mode == "particle_gp_hyper_pool":
                shared_spec = self._fit_shared_delta_hyper(X_hist, resid_shared, model_cfg)
                hyper_specs = self._build_hyper_pool(shared_spec)
                e.delta_state = ParticleSpecificGPDeltaState(
                    X_hist=X_hist,
                    Y_hist=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    hyper_specs=hyper_specs,
                )
                return

            if mode == "particle_basis":
                e.delta_state = ParticleSpecificBasisDeltaState(
                    X_hist=X_hist,
                    Y_hist=Y_hist,
                    theta_particles=theta_particles,
                    emulator=self._last_emulator,
                    rho=model_cfg.rho,
                    basis_kind=str(getattr(self.config, "particle_basis_kind", "rbf")),
                    num_features=int(getattr(self.config, "particle_basis_num_features", 8)),
                    lengthscale=float(getattr(self.config, "particle_basis_lengthscale", 0.25)),
                    ridge=float(getattr(self.config, "particle_basis_ridge", 1e-2)),
                    noise=max(float(getattr(self.config, "particle_basis_noise", model_cfg.delta_kernel.noise)), 1e-6),
                )
                return
        except Exception:
            pass

        # Safe fallback: revert to the previous expert-shared GP behavior.
        super()._reset_delta_for_expert(e, model_cfg)

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
        out = super().update_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            emulator=emulator,
            model_cfg=model_cfg,
            pf_cfg=pf_cfg,
            prior_sampler=prior_sampler,
            verbose=verbose,
        )

        anchor_e = self._anchor_expert_after_update()
        cur_stats = self._posterior_summary(anchor_e) if anchor_e is not None else None

        refresh_score = None
        gate_score = None
        cusum_triggered = False
        gate_triggered = False
        refresh_ok = False
        refresh_error = None

        if bool(out.get("did_restart", False)):
            self._cusum_stat = 0.0
            self._cusum_tau = int(self.t)
            self._gate_hits = 0
            self._cusum_prev_anchor_stats = cur_stats
        elif self.use_cusum and anchor_e is not None and cur_stats is not None:
            prev_stats = self._cusum_prev_anchor_stats
            if prev_stats is not None:
                try:
                    refresh_score = self._cusum_distance(prev_stats, cur_stats)
                    gate_score = math.sqrt(refresh_score)

                    if self.cusum_mode == "standardized_gate":
                        if gate_score > self.standardized_gate_threshold:
                            self._gate_hits += 1
                        else:
                            self._gate_hits = 0
                        if self._gate_hits >= self.standardized_gate_consecutive:
                            gate_triggered = True
                            refresh_ok = self._refresh_anchor_discrepancy(anchor_e, model_cfg)
                            if refresh_ok:
                                self._gate_hits = 0
                                self._cusum_stat = 0.0
                                self._cusum_tau = int(self.t)
                                out["restart_mode"] = "standardized_gate_refresh"
                                out["restart_message"] = "Standardized gate discrepancy refresh"
                    else:
                        self._cusum_stat += refresh_score
                        if self._cusum_stat > self.cusum_threshold:
                            cusum_triggered = True
                            refresh_ok = self._refresh_anchor_discrepancy(anchor_e, model_cfg)
                            if refresh_ok:
                                self._cusum_stat = 0.0
                                self._gate_hits = 0
                                self._cusum_tau = int(self.t)
                                out["restart_mode"] = "cusum_refresh"
                                out["restart_message"] = "CUSUM discrepancy refresh"
                except Exception as exc:
                    refresh_error = str(exc)
            self._cusum_prev_anchor_stats = cur_stats
        else:
            self._cusum_prev_anchor_stats = cur_stats

        out["particle_delta_mode"] = self._particle_delta_mode()
        out["cusum_enabled"] = bool(self.use_cusum)
        out["cusum_mode"] = self.cusum_mode
        out["cusum_threshold"] = float(self.cusum_threshold)
        out["cusum_recent_obs"] = int(self.cusum_recent_obs)
        out["cusum_score"] = refresh_score
        out["cusum_stat"] = float(self._cusum_stat)
        out["cusum_triggered"] = bool(cusum_triggered)
        out["standardized_gate_threshold"] = float(self.standardized_gate_threshold)
        out["standardized_gate_consecutive"] = int(self.standardized_gate_consecutive)
        out["gate_score"] = gate_score
        out["gate_hits"] = int(self._gate_hits)
        out["gate_triggered"] = bool(gate_triggered)
        out["refresh_triggered"] = bool(cusum_triggered or gate_triggered)
        out["cusum_refresh_ok"] = bool(refresh_ok)
        out["cusum_tau"] = int(self._cusum_tau)
        out["cusum_anchor_rl"] = int(anchor_e.run_length) if anchor_e is not None else None
        out["cusum_error"] = refresh_error
        return out
