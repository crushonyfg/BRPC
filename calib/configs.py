# =============================================================
# file: calib/configs.py
# =============================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Tuple, List, Protocol, Union, Sequence
import math
import torch


# ------------------- Configs -------------------
@dataclass
class DeltaKernelConfig:
    name: str = "rbf"  # or "matern52"
    lengthscale: Union[float, Sequence[float], torch.Tensor] = 1.0
    variance: float = 0.1
    noise: float = 1e-6  # nugget for numerical stability

'''
cfg1 = DeltaKernelConfig(name="rbf", lengthscale=1.0)  
# single scalar lengthscale

cfg2 = DeltaKernelConfig(name="rbf", lengthscale=[0.5, 1.0, 2.0])  
# per-dimension lengthscales

cfg3 = DeltaKernelConfig(name="matern52", lengthscale=torch.tensor([0.3, 0.7]))  
# tensor lengthscales
'''


@dataclass
class PFConfig:
    num_particles: int = 1024
    resample_ess_ratio: float = 0.5
    resample_scheme: str = "systematic"  # or "stratified", "multinomial"
    # move_strategy: str = "pmcmc"  # or "random_walk","liu_west", "laplace", "pmcmc", "none"
    move_strategy: str = "random_walk"
    # Liuâ€“West hyperparams
    liu_west_a: float = 0.90
    liu_west_h2: Optional[float] = None  # if None, derive from a via h^2 = 1-a^2
    # Random-walk and Laplace proposal scales
    random_walk_scale: float = 0.1
    laplace_alpha: float = 0.05
    laplace_beta: float = 1e-3
    laplace_eta: float = 0.01
    pmcmc_steps: int = 2


# @dataclass
# class BOCPDConfig:
#     def default_hazard(r: torch.Tensor) -> torch.Tensor:
#         """
#         Geometric hazard: h(r) = 1 / (Î» + r)
#         æœŸæœ› run-length = Î»
#         """
#         lam = 100.0  # æœŸæœ› 100 æ­¥å‘ç”Ÿä¸€æ¬¡å˜ç‚¹
#         return 1.0 / (lam + r)
#     # hazard: Callable[[torch.Tensor], torch.Tensor] = lambda r: torch.full_like(r, 0.01, dtype=torch.float64)  # h(r)
#     hazard: Callable[[torch.Tensor], torch.Tensor] = default_hazard
#     max_experts: int = 5  # keep top-k experts
#     max_run_length: int = 512  # truncation for run-length posterior (advisory)
#     restart_threshold: float = 0.8  # if P(CP) > threshold, trigger reset policy
#     log_space: bool = True
#     delta_refit_every: int = 0  # 0 means never
#     delta_refit_topk: int = 1  # 0 means no refit
#     use_restart: bool = True

# @dataclass
# class BOCPDConfig:
#     def default_hazard(r: torch.Tensor) -> torch.Tensor:
#         """
#         Geometric hazard: h(r) = 1 / (Î» + r)
#         æœŸæœ› run-length = Î»
#         """
#         lam = 100.0  # æœŸæœ› 100 æ­¥å‘ç”Ÿä¸€æ¬¡å˜ç‚¹
#         return 1.0 / (lam + r)
    
#     # âœ… æ–°å¢žï¼šé€‰æ‹©BOCPDæ¨¡å¼
#     bocpd_mode: str = "standard"  # "standard" æˆ– "restart"
    
#     hazard: Callable[[torch.Tensor], torch.Tensor] = default_hazard
#     max_experts: int = 10  # keep top-k experts
#     max_run_length: int = 512  # truncation for run-length posterior (advisory)
    
#     # âœ… Standard BOCPD ç›¸å…³é…ç½®
#     use_restart: bool = False  # ä»…ç”¨äºŽ standard mode
#     restart_threshold: float = 0.8  # ä»…ç”¨äºŽ standard mode
#     restart_small_r: int = 5  # ä»…ç”¨äºŽ standard mode
    
#     # âœ… R-BOCPD (restart_bocpd.py) ç›¸å…³é…ç½®
#     use_backdated_restart: bool = False  # False=Algorithm-2, True=Backdated
#     restart_margin: float = 0.05  # ç¨³å®šæ€§marginï¼Œé˜²æ­¢é¢‘ç¹restart
#     restart_cooldown: int = 10  # restartåŽçš„å†·å´æœŸï¼ˆæ­¥æ•°ï¼‰
    
#     log_space: bool = True
#     delta_refit_every: int = 1  # 0 means never
#     delta_refit_topk: int = 11  # 0 means no refit

@dataclass
class BOCPDConfig:
    """
    Configuration for Bayesian Online Change Point Detection (BOCPD)
    """
    # ==== æ ¸å¿ƒå‚æ•° ====
    hazard_lambda: float = 200   # <--- æ–°å¢žï¼šå¯è°ƒ Î» å€¼
    hazard_type: str = "geometric"  # <--- å¯é€‰ç±»åž‹ï¼šconstant, linear, weibull ç­‰

    def hazard(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute hazard function value h(r)
        """
        lam = self.hazard_lambda
        if self.hazard_type == "constant":
            return torch.full_like(r, 1.0 / lam)
        elif self.hazard_type == "linear":
            return torch.clamp(r / lam, max=1.0)
        elif self.hazard_type == "weibull":
            k = 1.5  # shape parameter
            return (k / lam) * ((r / lam) ** (k - 1))
        elif self.hazard_type == "geometric":
            return 1.0 / (lam + r)
        else:
            raise ValueError(f"Unknown hazard_type: {self.hazard_type}")

    # ==== å…¶ä»–å‚æ•° ====
    bocpd_mode: str = "standard"
    max_experts: int = 5
    max_run_length: int = 1000000
    use_restart: bool = False
    restart_threshold: float = 0.85
    restart_small_r: int = 5
    use_backdated_restart: bool = False
    restart_margin: float = 1
    restart_cooldown: int = 10
    log_space: bool = True
    delta_refit_every: int = 1
    delta_refit_topk: int = 11
    use_restart: bool = True

    restart_criteria: str = "rank_change" # "theta_test" or "rank_change"
    restart_theta_test: str = "energy" # or "credible" or "sw" or "energy"
    restart_cred_z: float = 2.0
    restart_cred_frac: float = 0.5
    restart_sw_proj: int = 32
    restart_theta_tau: float = 0.5
    controller_name: str = "none"  # for bocpd_mode == "single_segment": one of {"none", "sr_cs"}
    controller_stat: str = "surprise_mean"
    controller_cs_alpha: float = 0.01
    controller_cs_min_len: int = 2
    controller_cs_warmup_batches: int = 2
    controller_cs_max_active: int = 64
    controller_cs_clip_low: float = 0.0
    controller_cs_clip_high: float = 20.0
    controller_wcusum_warmup_batches: int = 3
    controller_wcusum_window: int = 4
    controller_wcusum_threshold: float = 3.0
    controller_wcusum_kappa: float = 0.5
    controller_wcusum_sigma_floor: float = 0.5





@dataclass
class ModelConfig:
    rho: float = 1.0
    sigma_eps: float = 0.05
    delta_kernel: DeltaKernelConfig = field(default_factory=lambda: DeltaKernelConfig(
        name="rbf",
        lengthscale=1.0,
        variance=0.01,  # âœ… è®¾ç½®ä¸º0ï¼Œç¦ç”¨delta
        # noise=1e-6,
        noise=1e-6
    ))
    emulator_type: str = "deterministic"  # or "gp"
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    use_discrepancy: bool = True
    refit_delta_every_batch: bool = True   # if False, skip delta-GP refit each batch (saves time when delta is unused)
    shared_delta_model: str = "gp"  # one of {"gp", "basis"}; affects shared offline refit path only
    delta_update_mode: str = "refit"  # one of {"refit", "online", "online_dynamic", "online_bpc", "online_bpc_exact", "online_bpc_exact_refithyper", "online_bpc_proxy_refithyper", "online_bpc_proxy_stablemean", "online_bpc_fixedsupport_exact", "online_inducing", "online_shared_mc_inducing", "online_shared_mc_inducing_refresh", "online_particle_mc_inducing", "online_particle_mc_inducing_refresh"}
    delta_online_min_points: int = 3
    delta_online_init_max_iter: int = 80
    delta_basis_num_features: int = 20
    delta_basis_prior_var_scale: float = 1.0
    delta_basis_fix_hyper: bool = False
    delta_dynamic_num_features: int = 20
    delta_dynamic_forgetting: float = 0.98
    delta_dynamic_process_noise_scale: float = 1e-3
    delta_dynamic_prior_var_scale: float = 1.0
    delta_dynamic_buffer_max_points: int = 256
    delta_bpc_lambda: float = 1.0
    delta_bpc_obs_noise_mode: str = "kernel"  # one of {"kernel", "sigma_eps"}
    delta_bpc_predict_add_kernel_noise: bool = True
    delta_inducing_num_points: int = 20
    delta_inducing_init_steps: int = 40
    delta_inducing_update_steps: int = 6
    delta_inducing_lr: float = 0.03
    delta_inducing_buffer_max_points: int = 256
    delta_inducing_learn_locations: bool = False
    delta_mc_num_inducing_points: int = 16
    delta_mc_num_particles: int = 8
    delta_mc_resample_ess_ratio: float = 0.5
    delta_mc_refresh_every: int = 0
    delta_mc_include_conditional_var: bool = True


@dataclass
class CalibrationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pf: PFConfig = field(default_factory=PFConfig)
    bocpd: BOCPDConfig = field(default_factory=BOCPDConfig)
