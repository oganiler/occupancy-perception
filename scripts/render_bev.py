# scripts/render_bev.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from occupancy.config import load_config
from occupancy.factory import build_adapter, build_grid_spec, build_voxelizer
from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.labels import OCC3D_CLASS_NAMES

from pathlib import Path

OUT = Path("/workspace/outputs")
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = [
    "#404040", "#ff9900", "#ff3d8c", "#3d8cff", "#0066cc", "#8c46ff",
    "#ff1f5e", "#ff5c5c", "#000000", "#00cc99", "#0044aa", "#ff66cc",
    "#cc99ff", "#7a5cff", "#00b894", "#8c8c8c", "#00994d", "#ffffff",
]

cfg = load_config()
spec = build_grid_spec(cfg)
adapter = build_adapter(cfg)
vox = build_voxelizer(cfg, spec)

s = adapter.get(adapter.sample_tokens()[200])
c = s.cloud
state, sem = vox.build(
    c.points_m, c.labels, c.sensor_origins_m,
    (~c.in_annotated_box) | c.from_reference,
)

# --- BEV of state: occupied wins over free, per column ---------------------
bev_state = np.full(state.shape[:2], UNOBSERVED, np.uint8)
bev_state[(state == FREE).any(axis=2)] = FREE
bev_state[(state == OCCUPIED).any(axis=2)] = OCCUPIED

# --- BEV of semantics: class of the LOWEST occupied voxel per column -------
occ = state == OCCUPIED
has = occ.any(axis=2)
first = occ.argmax(axis=2)
bev_sem = np.where(has, np.take_along_axis(sem, first[..., None], 2)[..., 0], 17)

# --- vertical slice through the ego row, showing the slab ------------------
slice_yz = state[:, state.shape[1] // 2, :]

fig, ax = plt.subplots(1, 3, figsize=(19, 6))

ax[0].imshow(bev_state.T, origin="lower", cmap=ListedColormap(
    ["#202020", "#5ec8ff", "#ff8c00"]), vmin=0, vmax=2)
ax[0].set_title("state: unobserved / free / occupied")

ax[1].imshow(bev_sem.T, origin="lower",
             cmap=ListedColormap(PALETTE),
             norm=BoundaryNorm(np.arange(19) - 0.5, 18))
ax[1].set_title("semantics, lowest occupied voxel")

ax[2].imshow(slice_yz.T, origin="lower", aspect="auto",
             cmap=ListedColormap(["#202020", "#5ec8ff", "#ff8c00"]),
             vmin=0, vmax=2)
ax[2].set_title("x-z slice at y = 0")

for a in ax[:2]:
    a.plot(state.shape[0] // 2, state.shape[1] // 2, "w+", ms=14, mew=2)

fig.suptitle(f"{s.scene_name}  {s.sample_token[:8]}")
fig.tight_layout()
fig.savefig("/workspace/outputs/bev.png", dpi=110)
print("wrote /workspace/outputs/bev.png")
print("classes:", [OCC3D_CLASS_NAMES[i] for i in np.unique(bev_sem)])

ids, cnt = np.unique(bev_sem[bev_sem != 17], return_counts=True)
for i, n in sorted(zip(ids, cnt), key=lambda t: -t[1]):
    print(f"{OCC3D_CLASS_NAMES[i]:<20} {n:,}")