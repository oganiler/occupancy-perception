# scripts/smoke_test.py
"""Verify the container has everything the pipeline needs."""
import sys


def main() -> int:
    import numpy as np
    import torch
    import matplotlib
    import nuscenes
    from omegaconf import OmegaConf

    print(f"python      {sys.version.split()[0]}")
    print(f"numpy       {np.__version__}")
    print(f"torch       {torch.__version__}  (cuda={torch.cuda.is_available()})")
    print(f"matplotlib  {matplotlib.__version__}")
    print(f"nuscenes    {nuscenes.__version__ if hasattr(nuscenes, '__version__') else 'installed'}")
    print(f"omegaconf   {OmegaConf.__module__ and 'installed'}")

    assert np.__version__.startswith("1."), "numpy 2.x will break nuscenes-devkit"

    x = torch.randn(2, 3, 4)
    print(f"torch forward check: {(x @ x.transpose(1, 2)).shape}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())