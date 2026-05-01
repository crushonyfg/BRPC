from pathlib import Path

import torch


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
SAMPLES = [
    ROOT / "figs" / "synthetic_cpd_suite_np1024_seed25_lambda2_raw_20260421_140855" / "raw_runs" / "sudden" / "sudden_Exact_BOCPD_seed_0_bs_20_mag_0.5000_seg_80.pt",
    ROOT / "figs" / "synthetic_cpd_suite_np1024_seed25_lambda2_raw_20260421_140855" / "raw_runs" / "sudden" / "sudden_Exact_wCUSUM_seed_0_bs_20_mag_0.5000_seg_80.pt",
    ROOT / "figs" / "synthetic_cpd_suite_halfrefit_np1024_seed25_20260424_103503" / "raw_runs" / "mixed" / "mixed_HalfRefit_seed_0_bs_20_drift_0.0080_jump_0.280_T_600.pt",
]

for path in SAMPLES:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"FILE={path.name}")
    print("cp_batches:", obj.get("cp_batches"))
    print("cp_times:", obj.get("cp_times"))
    hist = obj.get("restart_mode_hist")
    print("restart_mode_hist_unique:", sorted(set(hist)))
    print("restart_mode_hist_first20:", hist[:20])
    print("-" * 80)
