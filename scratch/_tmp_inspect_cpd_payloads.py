from pathlib import Path

import torch


ROOT = Path(r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration")
SAMPLES = [
    ROOT / "figs" / "synthetic_cpd_suite_np1024_seed25_lambda2_raw_20260421_140855" / "raw_runs" / "sudden" / "sudden_Exact_BOCPD_seed_0_bs_20_mag_0.5000_seg_80.pt",
    ROOT / "figs" / "synthetic_cpd_suite_halfrefit_np1024_seed25_20260424_103503" / "raw_runs" / "mixed" / "mixed_HalfRefit_seed_0_bs_20_drift_0.0080_jump_0.280_T_600.pt",
]


for path in SAMPLES:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"FILE={path}")
    if isinstance(obj, dict):
        print("TOP_KEYS=", sorted(obj.keys()))
        for k in sorted(obj.keys()):
            v = obj[k]
            if isinstance(v, dict):
                print(f"  {k}: dict keys={sorted(v.keys())[:30]}")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: {type(v).__name__} len={len(v)}")
            else:
                shape = getattr(v, "shape", None)
                print(f"  {k}: type={type(v).__name__} shape={shape}")
    print("-" * 80)
