from pathlib import Path

import torch


PATH = Path(
    r"C:\Users\yxu59\files\autumn2025\park\DynamicCalibration\figs\plantSim_cpd_ablation_seed10_bs20_np1024_20260424_225523\seed_runs\seed00\plantSim_results_mode2.pt"
)

obj = torch.load(PATH, map_location="cpu", weights_only=False)
print("TOP_LEVEL_METHODS:", sorted(obj.keys()))
for name in sorted(obj.keys())[:3]:
    rec = obj[name]
    print(f"\nMETHOD={name}")
    print("KEYS=", sorted(rec.keys()))
    for k in sorted(rec.keys()):
        v = rec[k]
        if isinstance(v, dict):
            print(f"  {k}: dict keys={sorted(v.keys())[:20]}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: {type(v).__name__} len={len(v)}")
        else:
            shape = getattr(v, "shape", None)
            print(f"  {k}: type={type(v).__name__} shape={shape}")
