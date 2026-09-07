# Ground truth generation — measured results

nuScenes v1.0-mini, all 404 samples across 10 scenes. Grid 200 x 200 x 16 at
0.4 m over [-40, -40, -1] to [40, 40, 5.4] m (Occ3D-nuScenes spec).
Keyframe-only aggregation (D-001), +/-4 keyframes, dynamic correction on,
`free_ray_stride=1` (D-006).

## Throughput

| Metric | Value |
|---|---|
| Samples | 404 |
| Wall clock | 1638 s (27 min) |
| Per sample | **4.06 s** |
| On disk | 19.4 MB |
| Compression | 13x |

Per-shard size ranges from 31 KiB (scene-0757) to 57 KiB (scene-0061), tracking
coverage: more supervised voxels means more entropy means a larger file.

## Voxel state distribution

| State | Mean | Min | Max |
|---|---|---|---|
| unobserved | 61.5% | 35.3% | 87.7% |
| free | 35.6% | 11.5% | 59.8% |
| occupied | 3.0% | 0.8% | 5.2% |

**38.6% of voxels carry supervision** on average, ranging from 12% to 65%
depending on the sample.

## Per-scene breakdown

Displacement is the ego's straight-line travel across the aggregation window
(+/-4 keyframes, ~4 s), derived from the `T_global_from_ego` stored in each
shard.

| Scene | n | unobs | range | occ | displacement | description |
|---|---|---|---|---|---|---|
| scene-0916 | 41 | 47.9% | 40–66% | 3.6% | 17.2 m | Parking lot, bicycle rack, bus, many peds |
| scene-1094 | 40 | 49.6% | 35–70% | 3.8% | 24.1 m | Night, after rain, many peds, jaywalker |
| scene-0796 | 40 | 50.6% | 41–66% | 4.1% | 45.4 m | Scooter, peds on sidewalk, bus, truck |
| scene-1077 | 41 | 52.3% | 44–60% | 4.0% | 47.5 m | Night, big street, high speed |
| scene-0061 | 39 | 54.8% | 47–68% | 3.2% | 17.7 m | Parked truck, construction, intersection |
| scene-0655 | 41 | 62.6% | 51–73% | 3.2% | 30.5 m | Parking lot, parked cars, bendy bus |
| scene-1100 | 40 | 66.8% | 58–71% | 2.0% | 0.2 m | Night, peds on sidewalk, crosswalk |
| scene-0103 | 40 | 67.2% | 62–75% | 3.2% | 22.3 m | Many peds, wait for turning car, bike rack |
| scene-0757 | 41 | 80.8% | 69–88% | 1.7% | 4.9 m | Arrive at busy intersection, **wait at intersection** |
| scene-0553 | 41 | 81.1% | **81–82%** | 0.9% | **0.0 m** | **Wait at intersection**, large truck, peds crossing |

## What drives coverage: ego parallax

Aggregation recovers coverage through **parallax, not sample count**. Nine
keyframes taken from nine different positions strike the ground at nine
different sets of beam radii, filling the gaps between rings. Nine keyframes
from a stationary vehicle are nine near-identical clouds: the same beams hit
the same radii, and no gap is ever filled.

Ego displacement across the aggregation window correlates with unobserved
fraction at **r = -0.673** (n = 404). The extremes are unambiguous:
scene-0553 travels 0.0 m and sits at 81% unobserved; scene-1077 travels
47.5 m and sits at 52%.

The signature is also visible in the *variance*. Every scene varies 10–35
points across its samples, except scene-0553, which varies by one point
(81–82%). A stationary ego sees the same scene from the same place at every
timestep, so its coverage is constant. Collapsed variance is the fingerprint of
a stationary sample.

Displacement explains roughly 45% of the variance (r^2 = 0.45); the remainder
is environmental. scene-1100 is also effectively stationary (0.2 m) yet reaches
67% rather than 81%, and scene-0916 achieves the best coverage in the split on
only 17.2 m of travel. Parallax is the dominant factor, not the only one.

### Correction to an earlier hypothesis

An initial reading of a 40-sample subset attributed the scene-level variation
to enclosure — the idea that open roads let rays travel further inside the
6.4 m slab before exiting. That hypothesis predicts intersections should score
*well*, since an intersection is more open than a narrow street. The measured
result is the opposite: the two worst-covered scenes are both described as
"wait at intersection". The enclosure explanation is not supported and has been
replaced by the parallax finding above.

## Aggregation window: secondary effect

Keyframe count is measurable but much smaller than the scene effect. Samples
near scene boundaries aggregate fewer keyframes after clamping:

| Aggregated keyframes | n | unobserved |
|---|---|---|
| 5 | 20 | 67.8% |
| 6 | 20 | 65.4% |
| 7 | 20 | 63.5% |
| 8 | 20 | 61.9% |
| 9 | 324 | 60.7% |

The full ramp is worth **7.1 points**. Scene means span **33 points**
(47.9% to 81.1%). Scene-level factors therefore account for roughly 4.7x the
variance that aggregation window size does.

## Why free space is bounded in absolute terms

Separate from the variance question: free space cannot approach 100% even for a
fast-moving ego, because the grid is a thin slab — 80 x 80 m wide but only
6.4 m tall. LIDAR_TOP sits at z = 1.84 m in the ego frame with a vertical field
of view of -30 to +10 degrees. Only beams between roughly -2.6 and +5 degrees
remain inside the slab for the full 40 m; the rest exit within metres. Free
space therefore concentrates into a disc around sensor height, visible as the
horizontal band in the x-z slice of `docs/figures/bev.png`.

## Occupied is stable; free is not

Occupied has a scene-mean range of 0.9–4.1%, and excluding the two stationary
outliers it clusters tightly at 3.2–4.1%. Free swings by a factor of five
(11.5–59.8% at sample level).

The two come from different mechanisms. Occupied is written by ray *endpoints*,
which reflect the physical density of surfaces in an urban scene — close to a
constant. Free is written by ray *traversal*, which depends on how far each
beam travels inside the slab and on how much parallax the window provides.

## Known limitations

- **Parallax bound.** For stationary samples, widening the aggregation window
  cannot improve coverage: the limitation is the absence of new viewpoints, not
  the number of clouds. Roughly 20% of the mini split is stationary or
  near-stationary. Only the label-propagation approach declined in D-001, or
  mesh-based hole filling, would help there. This is the precise condition
  under which D-001 costs coverage.
- **Ring gaps.** Individual LiDAR beams strike the ground at discrete radii;
  between the rings the surface is unobserved. Direct consequence of D-001.
  Visible as concentric arcs in the BEV figure.
- **Angular thinning at range.** 32 beams over 40 degrees gives ~1.25 degree
  resolution — 0.87 m at 40 m, more than two voxels. Free space becomes striped
  near the grid edge.
- **Supervision density varies 5x between samples.** The loss sees between 12%
  and 65% of voxels depending on the sample. Not currently compensated.
- **No camera visibility mask** (`gt.generate_camera_mask: false`), so the
  stored mask is LiDAR visibility only.

## Reproducing these numbers

```bash
# Full generation with per-sample statistics
python -m occupancy.cli gt 2>&1 | tee outputs/gt_full.log

# Per-scene coverage joined with nuScenes scene descriptions,
# plus camera and BEV figures for the extremes
python scripts/diag_scenes.py --scenes 2

# Ego displacement against coverage, computed from stored ego poses
python scripts/diag_motion.py
```

## DL pipeline throughput

Batch size 1, six cameras at 288 x 512, ResNet18 at stride 16, CPU only.

| Stage | Self CPU | Share |
|---|---|---|
| Backbone (`mkldnn_convolution`) | 0.614 s | 49% |
| View transform (`grid_sampler_2d`) | 0.398 s | 32% |
| Camera accumulate (`mul`, `add_`) | 0.134 s | 11% |
| Other | ~0.11 s | 8% |
| **Forward, end to end** | **1.51 s** | |

Reduced from 3.95 s by the channel reduction in D-010, chosen from this profile
rather than from where the cost appeared to be. The backbone is now the
dominant term.

Peak allocation in the view transform is 0.94 GB, against 3.66 GB before the
reduction.