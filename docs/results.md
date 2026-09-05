# Ground truth generation — measured results

nuScenes v1.0-mini. Grid 200 x 200 x 16 at 0.4 m over [-40, -40, -1] to
[40, 40, 5.4] m (Occ3D-nuScenes spec). Keyframe-only aggregation, +/-4
keyframes, dynamic correction on, `free_ray_stride=1`.

## Throughput

| Stage | Time |
|---|---|
| Adapter (load, transform, aggregate) | 0.07 s |
| Voxelization (DDA + endpoint pass) | 1.5–3.7 s, scene-dependent |
| Shard write (compress + save) | 0.01 s |
| **End to end** | **3.83 s/sample** |

Full mini split (404 samples): ~26 min, ~23 MB on disk at 11x compression.

## Voxel state distribution (40 samples, scene-0061 and scene-0103)

| State | Mean | Min | Max |
|---|---|---|---|
| unobserved | 55.2% | 47.3% | 74.6% |
| free | 41.6% | 23.6% | 49.3% |
| occupied | 3.1% | 1.8% | 3.7% |

Roughly **45% of voxels carry supervision** on average.

## What drives coverage variance

**Occupied is near-constant** (1.8–3.7%) while **free swings 2x** (23.6–49.3%).
Different mechanisms: occupied comes from ray endpoints, which reflect a
physical property of urban scenes; free comes from traversal, which depends on
how far rays travel before leaving the 6.4 m-tall slab.

**Scene geometry dominates keyframe count.** Across samples 5–39, where the
aggregation window is saturated at 9 keyframes, unobserved rises monotonically
from 47% to 68% as the ego drives from open surroundings into an enclosed
stretch — a 21-point swing at fixed keyframe count. The keyframe ramp at scene
boundaries moves it by only ~6 points (55% at kf=5 to 49% at kf=9). At the
scene change, samples 39 and 40 both have kf=5 yet differ by 7 points (68% vs
75%), scene-0103 being the tighter environment.

Practical consequence: coverage statistics quoted from a single sample are
unreliable. An early spot check of one scene-0757 sample gave 88% unobserved,
which the 40-sample distribution shows to be an outlier at the sparse end.

## Why free space is bounded by geometry

The grid is a thin slab: 80 x 80 m wide, 6.4 m tall. LIDAR_TOP sits at
z = 1.84 m with a vertical FOV of -30 to +10 degrees. Only beams between
roughly -2.6 and +5 degrees remain inside the slab for the full 40 m — under a
fifth of the FOV. The rest exit within metres. Free space therefore
concentrates into a disc around sensor height, visible as the horizontal band
in the x-z slice of `docs/figures/bev_scene0757.png`.

## Known limitations

- **Ring gaps.** Individual LiDAR beams strike the ground at discrete radii;
  between the rings the surface is unobserved. A direct consequence of D-001
  (no KNN label propagation or mesh reconstruction). Visible as concentric
  arcs in the BEV figure.
- **Angular thinning at range.** 32 beams over 40 degrees gives ~1.25 degree
  resolution — 0.87 m at 40 m, over two voxels. Free space becomes striped at
  the grid edge.
- **No camera visibility mask** (`gt.generate_camera_mask: false`).