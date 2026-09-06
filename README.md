# Occupancy-Aware Perception Pipeline

3D semantic occupancy ground-truth generation and a training-ready deep
learning pipeline, built on the nuScenes dataset.

Two independent halves joined by a file format:

    nuScenes -> adapter -> voxelizer -> [shard] -> Dataset -> model -> loss
             |------ GT generation ------|      |---- DL pipeline ----|
                    offline, once                    per epoch

Everything left of the shard imports `nuscenes-devkit`. Everything right of it
imports only `numpy`, `torch` and `PIL`. The devkit never enters a dataloader
worker, and the DL half is testable with no dataset mounted.

- **Design report:** [`docs/design_report.md`](docs/design_report.md)
- **Design decisions:** [`docs/decisions.md`](docs/decisions.md)
- **Measured results:** [`docs/results.md`](docs/results.md)
- **LLM assistance disclosure:** [`ASSISTANCE.md`](ASSISTANCE.md)

## Requirements

- Docker with Compose v2
- nuScenes **v1.0-mini** plus **nuScenes-lidarseg mini** (~4 GB)

No GPU. No Python installation on the host. The container is self-contained:
ImageNet weights for the backbone are baked in at build time, verified to run
with networking disabled.

## Dataset

Both archives extract into the **same** root directory:

```bash
mkdir -p ~/data/nuscenes && cd ~/data/nuscenes
wget https://www.nuscenes.org/data/v1.0-mini.tgz
wget https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2
tar -xf v1.0-mini.tgz
tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2
```

Expected layout:

```
~/data/nuscenes/
├── lidarseg/      <- from the lidarseg archive
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/     <- 13 JSON tables plus lidarseg.json
```

The dataset is an **external, read-only input**; it is not part of this
repository and is mounted rather than copied into the image.

## Quick start

```bash
docker compose build
docker compose run --rm occ python -m occupancy.cli gt --num-samples 5
docker compose run --rm occ python -m occupancy.cli infer --batch-size 1
```

If your nuScenes root is elsewhere, or your host UID is not 1000:

```bash
NUSCENES_ROOT=/your/path HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build
```

Both default sensibly, so most systems need neither.

## Commands

### Ground truth generation

```bash
# All 404 samples: ~27 min, ~19 MB
docker compose run --rm occ python -m occupancy.cli gt

# A subset, and a slice starting partway through
docker compose run --rm occ python -m occupancy.cli gt --num-samples 20
docker compose run --rm occ python -m occupancy.cli gt --start 79 --num-samples 20

# Any configuration key can be overridden on the command line
docker compose run --rm occ python -m occupancy.cli gt gt.window_keyframes=8
docker compose run --rm occ python -m occupancy.cli gt gt.free_sweep_stride=5
```

Writes one `.npz` per keyframe to `gt_shards/`, plus `config.snapshot.yaml` so
the directory records the settings that produced it.

### Inference and loss

```bash
docker compose run --rm occ python -m occupancy.cli infer --batch-size 2

# Fit one batch to verify gradient reaches every component (a wiring check,
# not training -- see docs/decisions.md D-009)
docker compose run --rm occ python -m occupancy.cli infer --overfit-steps 10
```

### Tests

```bash
docker compose run --rm occ pytest -v
```

Tests requiring nuScenes skip cleanly when the dataset is absent, so the suite
runs on a machine without the data.

### Verification and diagnostics

Each script backs a claim made in the report or a decision record, so the
measurements behind them can be reproduced rather than taken on trust.

| Script | Purpose | Supports |
|---|---|---|
| `smoke_test.py` | Container has torch, numpy, devkit; prints versions and asserts numpy < 2 | environment |
| `check_adapter.py` | Adapter loads a sample; point counts, label range, spatial extent | §2.3 |
| `check_gt.py` | End-to-end GT for one sample: timing, state histogram, in-grid fraction | §2.5 |
| `diag_adapter.py` | Ground plane, z-extremes by class, dynamic-correction effect | D-003, §2.3 |
| `diag_scenes.py` | Coverage per scene joined with nuScenes descriptions; camera + BEV figures | §3 |
| `diag_motion.py` | Ego displacement against coverage, from stored ego poses | §3, D-001 |
| `memory_layout.py` | Compression under three array layouts | D-005 |
| `render_bev.py` | BEV state, semantics, and x-z slice figures | §2.7 |

```bash
# environment
docker compose run --rm occ python scripts/smoke_test.py

# pipeline checks, single sample
docker compose run --rm occ python scripts/check_adapter.py
docker compose run --rm occ python scripts/check_gt.py

# analysis (diag_* accept config overrides)
docker compose run --rm occ python scripts/diag_adapter.py
docker compose run --rm occ python scripts/diag_adapter.py gt.window_keyframes=8
docker compose run --rm occ python scripts/diag_scenes.py --scenes 2
docker compose run --rm occ python scripts/diag_motion.py

# measurement behind D-005
docker compose run --rm occ python scripts/memory_layout.py

# figures
docker compose run --rm occ python scripts/render_bev.py
```

`diag_scenes.py` and `diag_motion.py` require shards to exist; run `gt` first.


## Configuration

All behaviour is driven by `configs/default.yaml`, validated against a typed
schema in `occupancy/config.py`. A mistyped key or a wrong type fails at load
rather than silently falling back to a default.

Defaults live only in the schema: components take required constructor
parameters, so none can be built in a state the configuration does not
describe.

| Key | Default | Effect |
|---|---|---|
| `grid.point_cloud_range` | `[-40,-40,-1, 40,40,5.4]` | Occ3D-nuScenes spec |
| `grid.voxel_size` | `[0.4, 0.4, 0.4]` | grid dims are derived, never configured |
| `gt.window_keyframes` | `4` | aggregation window is ±N keyframes |
| `gt.separate_dynamic` | `true` | box-local correction for annotated objects |
| `gt.free_ray_stride` | `1` | ray subsampling — see D-006 |
| `gt.free_sweep_stride` | `0` | non-keyframe free-space pass — see D-001 |
| `data.image_size` | `[288, 512]` | exact 16:9 from 900×1600 — see D-007 |
| `model.backbone.name` | `resnet18` | any torchvision ResNet |
| `loss.use_camera_mask` | `false` | see D-008 |

## Repository layout

```
occupancy/
├── config.py                  typed schema, validation, dotlist overrides
├── factory.py                 the only place config becomes objects
├── labels.py                  lidarseg -> Occ3D class mapping
├── metrics.py                 voxel mIoU
├── cli.py                     `gt` and `infer` entrypoints
├── data/
│   ├── nuscenes_adapter.py    devkit isolation, transform chain, aggregation
│   └── dataset.py             torch Dataset over shards; no devkit import
├── gt/
│   ├── grid.py                GridSpec; dims derived from range and voxel size
│   ├── shard.py               shard read/write
│   └── backends/
│       ├── base.py            VoxelizerBackend protocol
│       └── reference.py       Python voxelizer: DDA + endpoint passes
└── models/
    ├── backbone.py            parametric torchvision ResNet
    ├── view_transform.py      ViewTransform protocol + naive projection
    ├── occupancy_net.py       assembled network
    └── loss.py                masked cross-entropy

configs/default.yaml           all tunable behaviour
docs/                          design report, decisions, results, figures
scripts/                       diagnostics and figure generation
tests/                         unit tests
```

## What a shard contains

One `.npz` per keyframe, ~55 KiB compressed:

| Key | Shape | Meaning |
|---|---|---|
| `target` | (200, 200, 16) uint8 | 0–16 semantic, 17 free, **255 unobserved** |
| `image_paths` | (6,) | relative to the dataset root, not pixels |
| `camera_names` | (6,) | order of the camera axis |
| `intrinsics` | (6, 3, 3) | K at native resolution |
| `T_ego_from_cam` | (6, 4, 4) | extrinsics |
| `T_global_from_ego` | (4, 4) | ego pose; enables inter-frame motion later |
| `meta` | JSON | token, scene, timestamp, `num_keyframes`, point count |

255 is deliberately also `loss.ignore_index`, so the stored array is consumed
directly by cross-entropy with no separate masking step (D-005, D-008).

## Results

Full mini split, 404 samples, default configuration:

| | |
|---|---|
| Throughput | 4.06 s/sample, ~27 min total |
| Storage | 19.4 MB, 13× compression |
| Unobserved | 61.5% (35.3–87.7%) |
| Free | 35.6% |
| Occupied | 3.0% |

Coverage is bounded by ego parallax during the aggregation window
(r = −0.673, n = 404), not by scene enclosure. Full analysis in
[`docs/results.md`](docs/results.md).

## Scope

Per the assignment, **no model is trained**. The DL half is a training-ready
pipeline with a placeholder network. Verified by fitting a single batch for 20
steps: gradient reaches every parameter tensor and the loss falls from 2.99 to
1.18, with the starting value agreeing with ln(18) = 2.890 for an untrained
18-class classifier.

Measured cost of the alternative: 37.8 s per optimisation step at batch size 2
on CPU, so one epoch over 324 training samples would take 102 minutes.

## LLM assistance

Parts of this project were developed with LLM assistance. What was assisted and
what I reasoned about and implemented myself is recorded in
[`ASSISTANCE.md`](ASSISTANCE.md), together with the threshold I applied, errors
I found in assisted output, and analyses that were overturned by measurement.
