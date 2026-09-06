import numpy as np, io, time
from occupancy.gt.shard import read_shard
from pathlib import Path

from occupancy.config import load_config
cfg = load_config()
t = read_shard(sorted(Path(cfg.gt.output_dir).glob("*.npz"))[0]).target

def size(a, label):
    buf = io.BytesIO()
    t0 = time.perf_counter()
    np.savez_compressed(buf, x=np.ascontiguousarray(a))
    dt = time.perf_counter() - t0
    print(f"{label:<28} {buf.tell()/1024:6.1f} KiB   {t.size/buf.tell():5.1f}x   {dt:.2f}s")

size(t,                      "as stored (z fastest)")
size(t.transpose(2, 0, 1),   "z slowest (x fastest)")
size(t.transpose(2, 1, 0),   "z slowest, y fastest")
print(f"\nunique values: {np.unique(t).tolist()}")
print(f"255 fraction: {(t == 255).mean():.1%}")