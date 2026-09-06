# docs/decisions.md

## D-001: Keyframe-only sweep aggregation

**Status:** accepted

**Context.**
A single 32-beam LiDAR sweep yields ~35k points against a 200x200x16 = 640k
voxel grid, touching at most ~5% of it. Aggregating multiple sweeps is
therefore required to produce usable occupancy ground truth.

nuScenes-lidarseg provides per-point semantic labels for keyframes only:
404 lidarseg records against 31,206 sample_data records in v1.0-mini.
Non-keyframe sweeps carry geometry but no semantic label.

**Options.**
- A. Aggregate keyframes only (2 Hz). Every point carries a ground-truth label.
- B. Aggregate all sweeps (20 Hz) and propagate labels to unlabelled points
     by k-nearest-neighbour, as in Occ3D. ~10x the points, but ~89% of them
     would carry inferred rather than measured labels.

**Decision.** Option A.

**Rationale.**
Ground truth quality is the property the entire downstream pipeline is
supervised against. Option B would introduce a heuristic into the ground
truth itself, making it impossible to distinguish model error from label
error when evaluating. Option A trades point density for label fidelity:
every occupied voxel's semantic label traces to a measured annotation.

Coverage is recovered instead through the aggregation window (+/-4 keyframes,
+/-2 s) and through static/dynamic separation, which allows a wider window
without smearing moving objects.

**Consequences.**
- Lower point density than the Occ3D reference implementation.
- No KNN or spatial index needed in the GT path.
- Option B remains available behind the same aggregation interface; it is
  listed as a future extension.

**Revisit if.** Free-space and occupied coverage prove too sparse to train
against, measured as the fraction of voxels remaining `unobserved`.

**Measured boundary of this decision.**
Aggregation recovers coverage through ego parallax, not through sample count.
Across 404 samples, ego displacement over the aggregation window correlates
with unobserved fraction at r = -0.673. Scenes where the ego is stationary
(scene-0553 at 0.0 m, scene-0757 at 4.9 m) sit at 81% unobserved, while scenes
with 45+ m of displacement sit near 51%.

For a stationary sample, nine keyframes are nine near-identical point clouds:
the same beams strike the same ground radii, so the ring gaps between beams
are never filled. Widening the window cannot help, because the limitation is
parallax rather than the number of clouds.

This is the precise condition under which Option A costs coverage. Roughly 20%
of the mini split is stationary or near-stationary (waiting at intersections),
and for those samples only the label-propagation approach of Option B, or
mesh-based hole filling, would improve ground-truth density.

Displacement explains about 45% of the variance in coverage; the remainder is
environmental, since scene-1100 is also stationary (0.2 m) yet reaches 67%
rather than 81%.

**Refinement: non-keyframe sweeps for free space only.**

The keyframe-only restriction is driven by *labels*, and labels are needed only
at ray endpoints. A traversed voxel is class 17 (`free`) regardless of what the
beam eventually struck. Non-keyframe sweeps are therefore usable for free-space
determination even though nuScenes-lidarseg annotates keyframes only.

This targets the measured bottleneck directly. Coverage is parallax-limited
(r = -0.673 against ego displacement over the window), and at 20 Hz the ego is
sampled roughly every 0.5 m rather than every 5 m, so the inter-beam ring gaps
are filled at far finer granularity. Free space is 35.6% of the grid and
carries essentially all of the coverage variance; occupied is 3.0% and is
already saturated from keyframes alone.

Implemented as a third pass, ordered last:

    pass 1  free space from keyframe rays          -> FREE
    pass 2  keyframe endpoints and semantic votes  -> OCCUPIED
    pass 3  free space from non-keyframe rays      -> FREE, only where UNOBSERVED

Endpoints of non-keyframe rays are simply not cast. No fourth state, no
unlabelled-occupied voxel, no additional sentinel.

The `UNOBSERVED`-only guard in pass 3, and its placement after pass 2 rather
than before pass 1, prevent a specific error. A moving vehicle occupies volume
at t_ref that was empty air two seconds earlier, so a non-keyframe beam passed
straight through it. Marking those voxels free would contradict the
box-corrected occupancy that pass 2 writes there (D-003). Because LiDAR returns
only from an object's near surface, pass 2 marks a shell; running pass 3 first
would leave free space inside that shell — a hollow vehicle. The guard makes
pass 3 strictly additive: it can only fill voxels that no keyframe measurement
determined, and can never contradict one.

Residual limitation: voxels inside a moving object's volume that no keyframe
endpoint reached remain incorrectly free. This is bounded by the object's
surface sampling density and has not been quantified.

**Attribution.** This refinement was proposed by the author. The ordering guard
was added in review. See ASSISTANCE.md.

**Measured, and its boundary.Result Of The Refinement**

| Scene | window displacement | unobserved, stride 0 | stride 5 | gain | s/sample |
|---|---|---|---|---|---|
| scene-0061 | 17.7 m | 49.9% | 43.9% | 6.0 pts | 3.8 -> ~11 |
| scene-0553 | 0.0 m | 81.2% | 80.0% | 1.2 pts | 7.7 -> 22.5 |

The initial expectation was that pass 3 would help *most* where parallax is
absent. The measurement refutes this: the stationary scene gained five times
less at three times the cost.

The reason refines the mechanism. Non-keyframe sweeps are taken from wherever
the ego was at that instant. For a stationary vehicle they originate at the
same position, so the same 32 beams strike the same 32 ground radii and the
inter-beam gaps are geometrically unreachable. Pass 3 supplies additional
*timesteps*, which become additional *viewpoints* only in proportion to ego
motion. It multiplies existing parallax; it cannot create parallax.

Cost is roughly constant (~3x the ray count) while benefit scales with motion,
so the return is worst exactly where coverage is worst.

This independently confirms the stationary limitation recorded above: for
samples with no ego motion, no amount of additional ray casting helps. Only
label propagation (Option B) or mesh-based hole filling would.

**Shipped configuration:** `free_sweep_stride: 0`. The capability is
implemented and covered by tests (`test_pass3_only_fills_unobserved`,
`test_pass3_fills_untouched_voxels`), and left disabled because a ~2 hour
regeneration for ~4 points averaged across the split was not the best use of
the remaining time. Enabling it is a one-line configuration change.

**Guard verified empirically.** The `occupied` fraction is byte-identical
across all strides in both scenes (3.3% and 1.0% respectively, including min
and max), confirming that pass 3 never overwrites an occupied voxel across
~12.8 million voxels — the property the unit tests pin on five.

## D-002: Occ3D-nuScenes class set, derived from the devkit's coarse mapping

**Status:** accepted

**Context.**
nuScenes-lidarseg annotates points with 32 fine-grained classes
(`human.pedestrian.child`, `vehicle.bus.bendy`, `static.other`, ...).
The Occ3D-nuScenes benchmark uses 18: indices 0..16 semantic, plus index 17
`free`. A mapping from the fine set to the target set is therefore required.

**Options.**
- A. Define a project-specific class set and hand-write the 32 -> N table.
- B. Adopt the Occ3D-nuScenes 18-class set and hand-write the 32 -> 18 table.
- C. Adopt the Occ3D-nuScenes 18-class set and derive the table from the
     devkit's `LidarsegClassMapper`.

**Decision.** Option C.

**Rationale.**
Adopting the Occ3D class set (over Option A) makes any future numbers directly
comparable with published results, and means the grid specification and the
label space come from the same reference rather than being independently
invented.

Deriving the table (over Option B) rather than transcribing it removes a class
of silent error. `LidarsegClassMapper.get_fine_idx_2_coarse_idx()` collapses
the 32 fine classes onto indices 0..16, and those indices coincide exactly,
slot for slot, with Occ3D classes 0..16. The devkit's index 0 is named
`ignore`; Occ3D names the same slot `others`, and the fine classes routed
there by the devkit (noise, animal, emergency vehicles, wheelchair, stroller,
personal mobility, debris, pushable/pullable, bicycle rack, `static.other`,
`vehicle.ego`) are the same set Occ3D treats as `others`.

A transcribed 32-entry table would be correct on the day it was written and
unverifiable thereafter; a derived one cannot drift from the devkit's own
definition of the classes.

**Consequences.**
- No hand-maintained class table in this repository.
- The mapping is materialised once per dataset as a (32,) uint8 lookup array
  and applied by array indexing, not per-point dictionary lookup.
- Class 17 (`free`) is *not* reachable from any point label. It is assigned
  exclusively by ray traversal in the voxelizer. `build_fine_to_occ3d_lut`
  asserts that the table's maximum value is 16, so a future change that
  violated this would fail loudly rather than corrupt the ground truth.
- `unobserved` is deliberately absent from the class set: it is a mask state,
  not a class, and is never predicted by the model. See D-008 (loss masking)
  when that decision is recorded.
- Adds a dependency on `nuscenes.eval.lidarseg.utils`, an evaluation-side
  module, from the GT generation path. Accepted: the alternative is
  duplicating its content.

**Verification.**
The claimed index-for-index correspondence with Occ3D is asserted, not proven,
by this document. It is checked once against the published Occ3D class list
when the first labelled grid is rendered; a silent off-by-one here would
propagate into every downstream label. A unit test pins the resulting table
so that a devkit upgrade cannot change it unnoticed.

**Revisit if.** The 16-class collapse proves too coarse for the intended
application. A ground robot may, for example, need `pushable_pullable` and
`debris` as distinct traversability-relevant classes rather than folded into
`others`.

**Verification.** Confirmed. Rendering the ground-truth semantics in BEV for
scene-0757 produces a coherent street scene: `driveable_surface` forms a
connected drivable region around the ego (1,005 columns, the largest class),
bordered by `sidewalk` and `terrain`, with `manmade` (988) on the building
facades. Vehicle classes appear as compact clusters on the road surface, and
small objects (`pedestrian`, `traffic_cone`) survive voxelization at 0.4 m.
The devkit-derived index mapping is therefore correct against rendered
geometry, not only against the published class list.

A unit test pins the resulting (32,) lookup table so a devkit upgrade cannot
change it unnoticed.

## D-003: Ray-casting policy for box-corrected dynamic points

**Status:** accepted

**Context.**
Free space in the occupancy grid can only come from ray traversal: a voxel is
`free` because a beam passed through it. Occupied voxels come from beam
endpoints. The voxelizer therefore needs, per point, an origin and an endpoint.

D-001 aggregates +/-4 keyframes. For points on annotated objects, the adapter
applies a box-local correction: a point observed on an object at time t_i is
placed where that object sits at t_ref. This is what prevents a vehicle at
10 m/s from smearing 40 m across the grid.

That correction breaks the ray. The beam was fired from origin O to the
observed surface A. After correction the point is at B. O->A is what was
physically measured; B is where the geometry belongs at t_ref. For an
approaching vehicle B lies between O and A, so the measured ray passes
directly through the corrected object.

**Options.**
- A. Cast O->B. Asserts free space along a path that was partly occluded by
     the object at A when the measurement was taken.
- B. Cast O->A and mark A occupied. Leaves a ghost at a position the object
     has since vacated.
- C. Cast O->A for free space only, insert B as occupied separately. The free
     ray traverses B, erasing the object the correction just reconstructed.
- D. Cast no ray for corrected points. Use them only to mark B occupied.
     Free space comes from static points and from the reference sweep.

**Decision.** Option D, with an exception for the reference sweep.

Per-point ray policy:

| Point source                            | Free ray | Marks occupied |
|-----------------------------------------|----------|----------------|
| Static, any keyframe in the window       | yes      | yes            |
| Inside an annotated box, reference sweep | yes      | yes            |
| Inside an annotated box, other keyframes | **no**   | yes            |

The reference sweep is exempt because at t_ref the correction is the identity:
observed and corrected positions coincide, so the measured ray remains valid.

**Rationale.**
Options A, B and C each introduce a claim the sensor never made. Option D
declines to claim anything about space it cannot vouch for. Voxels that would
have been marked free by a discarded ray remain `unobserved` -- which is the
honest answer, and which the visibility mask already exists to express.

This is consistent with D-001: where the data does not support a determination,
prefer `unobserved` over an inferred value. Ground truth quality is the
property the entire pipeline is supervised against.

**Precedence.** Occupied wins over free. A voxel receiving both a traversal
and an endpoint is `occupied`: a return is direct evidence of a surface,
whereas a traversal is evidence only along one line of sight and may be
contradicted by another. Without this rule the result would depend on ray
processing order, which must not happen -- the voxelizer is required to be
deterministic.

**Consequences.**
- Reduced free-space coverage in regions dominated by moving objects,
  typically the road surface ahead of and behind traffic. Those voxels stay
  `unobserved` rather than being wrongly marked `free`.
- Requires the per-point flags `in_annotated_box` and `from_reference` that
  the adapter already emits. The adapter supplies facts; this policy consumes
  them. No adapter change is needed.
- The policy is a voxelizer-level concern and must be identical across the
  Python and C++ backends, so the parity test covers it implicitly.

**Measurement.** Report the fraction of voxels in each state, split by whether
dynamic correction is enabled. The expected signature is a modest rise in
`unobserved` and a fall in `free`, with `occupied` roughly unchanged. A large
change in `occupied` would indicate the policy is being applied incorrectly.

**Revisit if.** `unobserved` fraction rises far enough that too few voxels
carry supervision. The fallback is to cast free rays only up to a conservative
margin short of the corrected position, accepting some inference in exchange
for coverage.

**Why t_ref is the correct correction target.**
A nuScenes sample is a hardware-synchronised snapshot: camera exposure is
triggered as the top LiDAR sweeps across each camera's field of view. The
reference keyframe is therefore the timestamp of the images the model
consumes. Correcting dynamic objects to any other time in the window would
place them systematically ahead of or behind their appearance in the input
images, making the target unlearnable.

Residual limitation: the six cameras fire in sequence across one LiDAR
rotation (bounded by 50 ms at 20 Hz), so a single grid cannot be exactly
simultaneous with all six views. At 10 m/s this is roughly one voxel of
displacement. Not modelled; noted.

## D-004: Amanatides-Woo for ray traversal

**Status:** accepted

**Context.**
Free space is the only occupancy state that cannot be derived from the points
alone: a voxel is `free` because a beam passed through it. The voxelizer must
therefore enumerate, for each ray, the set of voxels between sensor origin and
returned point. On scene-0757 this is ~102,000 rays per sample and dominates
the runtime.

Correctness here is not cosmetic. Free space is 11.6% of the grid and is what
the loss is computed against for non-occupied voxels. An enumeration that
misses voxels leaves holes in the ground truth that would still look
approximately right in a BEV render.

**Options.**
- A. Sample the ray at fixed intervals of t and floor each sample to a voxel.
- B. 3D Bresenham line drawing.
- C. Amanatides-Woo voxel traversal.

**Decision.** Option C.

**Rationale.**
Option A is approximate by construction. Voxel boundary crossings are
irregularly spaced along a ray, so no fixed step size is simultaneously
complete (misses no crossed voxel) and non-redundant (revisits none). Too
coarse leaves stochastic holes in free space; too fine wastes work.

Option B solves a different problem. Bresenham produces a connected line of
voxels, taking one voxel per major-axis step. Where a ray crosses a
perpendicular boundary partway along a step, the ray is genuinely inside two
voxels but Bresenham reports one. The omissions are systematic along diagonal
rays, not random.

Option C is exact: it returns every voxel whose volume the ray intersects, in
order, with no duplicates and no gaps. The authors describe it as a variant of
the DDA line algorithm that permits simpler traversal "at the expense of more
voxels" -- precisely the trade-off this application wants, since the extra
voxels are exactly the ones a line algorithm wrongly omits.

Cost per voxel is O(1): the published figure is two floating-point comparisons
and one floating-point addition per step. All division is hoisted into the
per-ray setup.

**Implementation notes.**
The traversal runs in continuous *voxel* coordinates rather than metres. This
sets the voxel size to 1, so `t_delta[k]` reduces to `1 / |d_k|` with no voxel
size threaded through the loop, and it makes the integer part of a coordinate
the voxel index directly, so traversal and indexing cannot disagree about
where a boundary is.

The direction vector is `end - origin`, unnormalised, so the ray parameter t
runs 0 to exactly 1.0. Termination is `t_max[axis] > 1.0`, an exact
comparison requiring no epsilon. A normalised direction compared against a
metric ray length would require a tolerance, and tolerances need tuning.

The endpoint voxel is excluded from the free pass; pass 2 claims it as
occupied (D-003).

**Consequences.**
- The free-space pass is an inherently sequential per-ray loop and accounts
  for essentially all of the 1.54 s/sample runtime. Pass 2 is vectorised and
  already runs at NumPy speed. Any future native backend should therefore
  target the traversal, not the endpoint pass.
- `free_ray_stride` exists as a coverage-for-runtime knob, since adjacent rays
  traverse nearly identical voxel sets. Currently 1.
- Behaviour is pinned by tests: axis-aligned traversal, the no-free-ray path,
  out-of-bounds cropping, and bit-exact determinism across runs.

**Note on what was strided.** `free_ray_stride` subsamples the flat aggregated
point array, not LiDAR beams. Points arrive in acquisition order across nine
concatenated sweeps, already filtered by range and by the D-003 dynamic mask,
so striding removes rays scattered arbitrarily across beams, azimuths and
sweeps rather than dropping whole beams. This is why free space perforates into
speckle rather than thinning into a coarser but still-solid ring pattern.

Structured beam-level subsampling would likely degrade more gracefully. It was
not implemented: nuScenes does not publish per-point ring indices, so beams
would have to be recovered by clustering elevation angles, and the parameter
has been rejected on other grounds regardless.  

**Attribution.** The algorithm is published prior work, cited above and in
ASSISTANCE.md. The implementation in `occupancy/gt/backends/reference.py` is
this project's own.

## D-005: Shard format — single target array with a sentinel

**Status:** accepted

**Context.**
Ground truth must persist between GT generation (offline, once) and the DL
pipeline (per epoch). The voxelizer produces two arrays: `state`
(unobserved/free/occupied) and `semantics` (Occ3D class per voxel). The
question is what lands on disk.

**Options.**
- A. Store both arrays. Self-documenting, but partly redundant: `semantics`
     already encodes occupied (0..16) and free (17); only `unobserved` is not
     representable.
- B. Store one array with 255 as an `unobserved` sentinel.

**Decision.** Option B. `target` is (200, 200, 16) uint8: 0..16 semantic
class, 17 free, 255 unobserved.

**Rationale.**
The stored array *is* the cross-entropy target. `ignore_index=255` consumes it
directly, with no separate masking step to write, test, or get wrong. The
LiDAR visibility mask is implicit (`target != 255`) rather than a second
stored array that could drift out of sync with the first.

It also makes D-002's "unobserved is a mask state, not a class" structural:
255 lies outside the class range by construction, so no code path can treat it
as a class.

`config.py` asserts `loss.ignore_index == UNOBSERVED_LABEL` at load, so a
mismatch fails at startup rather than silently producing a wrong loss.

**Storage layout: measured and left alone.**
Compression was measured rather than assumed. `savez_compressed` gives 11x
(≈55 KiB/shard) at 0.01 s per shard, so compression is effectively free and
turns ~250 MB into ~25 MB for the mini split.

The array is stored in its natural memory order, where z varies fastest.
Transposing so that z varies slowest improves compression to 16.1x — a 26%
reduction, or ~5 MB across the full mini split. Rejected: it introduces a
permanent axis-order asymmetry between memory and disk, and a single forgotten
transpose would silently produce a rotated grid. 5 MB does not buy that risk.

**Contents.** Each shard holds the target grid, relative image paths (not
pixels), camera names, intrinsics, ego-from-camera extrinsics, the global-from-
ego pose, and a JSON metadata blob (sample token, scene, timestamp,
`num_keyframes`, point count).

Image paths are stored relative to the dataset root so a shard directory is
portable; the Dataset rejoins them with its own configured root. The ego pose
is stored so inter-sample ego motion can be derived later, which is what
temporal fusion requires — cheap now, expensive to retrofit.

`config.snapshot.yaml` is written alongside the shards, so a shard directory
records the window size, stride and grid spec that produced it.

**Consequences.**
- The shard is the architectural seam. Everything upstream imports `nuscenes`;
  everything downstream imports only `numpy` and `torch`. The devkit never
  enters a dataloader worker, and the DL pipeline is testable against synthetic
  shards with no dataset mounted.
- Any consumer wanting the three-state view derives it: `==255`, `==17`,
  `else occupied`.
- `read_shard` uses `allow_pickle=False`. An npz requiring pickle to load can
  execute arbitrary code, and nothing stored here needs it.

  ## D-006: free_ray_stride left at 1

**Status:** accepted

**Context.**
Adjacent LiDAR rays traverse nearly identical voxel sets, so the free-space
pass is highly redundant. `free_ray_stride` subsamples rays to trade coverage
for runtime.

**Measurement (40 samples).**

| stride | s/sample | disk   | compression |
|--------|----------|--------|-------------|
| 1      | 3.83     | 2.3 MB | 11x         |
| 4      | 1.86     | 2.7 MB | 10x         |

**Decision.** Leave at 1.

**Rationale.**
Two results argue against striding.

Quartering the rays gives only a **2.1x** speedup, not 4x. Traversal dominates
runtime but does not monopolise it; the adapter and the vectorised endpoint
pass form a fixed floor. Even an infinitely fast traversal would cap the
end-to-end gain near 2x.

More surprisingly, striding produces **larger** files (2.3 → 2.7 MB). Fewer
rays do not shrink the free region uniformly — they perforate it. Solid blocks
of `free` become free voxels speckled through unobserved ones, because the
adjacent rays that would have filled the gaps are skipped. Speckle has higher
local entropy than either solid region, so it compresses worse.

So striding costs coverage *and* disk, buying only 2x time on a step that is
not a bottleneck at this dataset size (404 samples ≈ 27 minutes).

**Consequences.** The parameter remains in the configuration as a knob for
larger datasets, where the absolute time saving would matter more.

## D-007: Aspect-preserving image resize over the benchmark crop convention

**Status:** accepted

**Context.**
nuScenes cameras are 1600 x 900. Six images per keyframe at full resolution is
too expensive for CPU-only inference, so the DL pipeline must downscale. Any
change to image geometry must be matched by a corresponding change to the
camera intrinsics, because the view transform projects 3D voxel centres into
image coordinates using K. An unadjusted K produces a projection that samples
the wrong pixels: nothing raises, nothing crashes, and the output looks
approximately correct.

**Options.**
- A. Crop-and-resize to 256 x 704, the BEVDet-family convention. Resize by
     704/1600 = 0.44, then crop 140 rows from the top to keep the bottom 256.
- B. Aspect-preserving resize to 288 x 512. Scale 512/1600 = 0.32 exactly,
     giving 900 x 0.32 = 288 exactly.

**Decision.** Option B, 288 x 512.

**Rationale.**
Option A requires two intrinsic corrections: focal length and principal point
scale with the resize, and the principal point shifts again by the crop offset.

    K[0,0] *= s ; K[1,1] *= s
    K[0,2] *= s ; K[1,2] *= s
    K[1,2] -= crop_top

Option B requires one:

    K[:2, :] *= s

Halving the number of independent corrections halves the opportunities for a
silent projection error, which is the dominant failure mode in this part of the
pipeline. The crop offset in particular is a constant that must agree between
the image transform and the intrinsic adjustment, with nothing enforcing that
agreement.

The chosen resolution is not arbitrary. 0.32 maps 1600 -> 512 and 900 -> 288
with no rounding, preserving the native 16:9 aspect exactly, so no distortion
is introduced. Both dimensions are divisible by 32, so standard backbone stride
sequences divide cleanly and no padding is needed.

Option B is also 18% cheaper: 147,456 pixels per image against 180,224.

**What is given up.**
The crop in Option A is not merely a convention. It removes sky, which cannot
contain an occupied voxel given the grid's z range of [-1, 5.4] m, so Option A
spends its pixel budget more usefully. Option B retains those rows and wastes
some capacity on them.

That trade-off favours Option A when training, where pixel efficiency
translates into accuracy per unit compute. It does not apply here: no model is
trained (per the assignment brief), and no results are compared against
published benchmark numbers, so conformance to the BEVDet convention buys
nothing. Under those conditions the reduction in silent-failure surface is
worth more than the pixel efficiency.

**Consequences.**
- `data.image_size: [288, 512]` (H, W) in configuration.
- Intrinsic adjustment is a single scale, applied in the Dataset alongside the
  resize, so the two cannot be applied independently of one another.
- A unit test asserts that a known 3D point projects to the same normalised
  image location before and after resize, which is the property that actually
  matters and which no amount of shape checking would catch.
- Switching to Option A later means adding the crop and its principal-point
  offset in one place; nothing downstream depends on the resolution.

**Revisit if.** The pipeline is ever trained, or results are to be compared
against published Occ3D-nuScenes numbers. In either case Option A becomes the
correct choice.

## D-008: Loss masking policy and the camera visibility mask

**Status:** accepted

**Context.**
The stored target uses 255 for voxels no LiDAR ray determined (D-005). The loss
must decide what to do with them, and separately whether to restrict
supervision further to voxels the *cameras* could see.

Occ3D provides two masks. `mask_lidar` marks voxels observed by LiDAR during
generation; `mask_camera` marks voxels visible to the cameras at the reference
timestamp. The benchmark evaluates using the camera mask, excluding unobserved
voxels from the mIoU.

**Decision.**
1. LiDAR-unobserved voxels are excluded from the loss via `ignore_index=255`.
2. The camera visibility mask is **not generated** (`gt.generate_camera_mask:
   false`), and therefore not consumed (`loss.use_camera_mask: false`).

**Rationale for (1).**
A voxel no ray reached has no measured label. Supervising against a guess would
put inference into the ground truth, which D-001 exists to avoid. `ignore_index`
means those positions produce no gradient — the forward pass still emits a
class there, but nothing corrects it. Unobserved is a mask state, not a class:
the head has 18 logits and none of them means "unknown" (D-002).

Because the stored sentinel is exactly `ignore_index`, the target array is
consumed by `F.cross_entropy` with no separate masking step. `config._validate`
asserts the two constants agree, so a mismatch fails at load rather than
silently producing a wrong loss.

**Rationale for (2) — and why it is not simply an omission.**
The camera mask is not a free improvement. SparseOcc report that training two
BEVFormer variants with and without the visible mask gives the masked version
15 points *higher* voxel mIoU but 1 point *lower* RayIoU, and attribute this to
the mask leaving the region behind each surface unsupervised, so the model
fills it with duplicated predictions and produces a thicker, closer surface.

That is a supervision-policy trade-off, not a data-format detail:

- mask on  -> visible-surface reconstruction. Clean supervision, high voxel
              mIoU, no amodal reasoning, degraded depth at range.
- mask off -> scene completion. The model must infer the occluded volume,
              which is arguably why occupancy is preferred over bounding boxes.

For a ground robot, conservative estimation of occluded volume plausibly
matters more than voxel-level mIoU. The decision is therefore left open rather
than settled by default, and the configuration flag exists so it can be
measured once a model is trained.

**Consequences.**
- Supervision density varies by sample: measured at 55.5% and 63.5% on two
  validation batches, and 38.6% mean across the full split (D-005 results). The
  loss is normalised over supervised voxels only, so this changes the effective
  batch size rather than the loss scale.
- The stored `target != 255` is the LiDAR visibility mask, implicit rather than
  stored separately.
- Comparing masked and unmasked loss at initialisation is uninformative: with
  untrained weights the model is uniformly wrong, so the two differ by under
  0.1% and the sign of the difference varies between runs. The comparison only
  becomes meaningful after the model has fitted something.

**Revisit if.** A model is trained. Then generate the camera mask and measure
both policies against RayIoU rather than voxel mIoU, since the latter cannot
observe the surface-thickening failure.


## D-009: Placeholder model architecture

**Status:** accepted

**Context.**
The brief states that no model need be trained; the deep-learning criterion is
pipeline design, not model quality. The network must nonetheless be a real
network: it must consume the batch contract, produce logits matching the target
grid, and support a loss with gradients reaching every component.

**Decision.** Four injected stages, none of them fixed:

    backbone  ->  view transform  ->  voxel encoder  ->  head

- **backbone**: torchvision ResNet, parametric in name, pretrained flag and
  output stride. Default `resnet18`, ImageNet weights, stride 16. Shared across
  all six views by folding the camera axis into the batch.
- **view transform**: naive backward projection.
- **voxel encoder**: two 3D convolutions.
- **head**: 1x1x1 convolution to 18 classes.

**Backward projection over forward projection.**
Forward projection (Lift-Splat-Shoot) takes a pixel and places its feature in
3D, which requires predicting *where* — a depth distribution per pixel. That is
the ill-posed part of monocular vision.

Backward projection takes a voxel centre and finds its pixel. No depth
prediction is needed, because the 3D location is chosen rather than inferred:
the grid is the hypothesis space, and "if something were here, which pixel
would show it?" has one exact answer given K and the extrinsics. Twenty lines of
geometry instead of a learned depth head.

The cost is depth ambiguity: two voxels at 10 m and 30 m along the same ray
project to the same pixel and sample the same feature. This is the documented
weakness of the backward family — dense BEV features, but points along a ray
acquire indistinctive features. FB-OCC combines both directions for this
reason.

Both alternatives implement the same `ViewTransform` protocol, so replacing the
naive transform touches the factory and nothing else.

**Backbone choice.**
ResNet18 at stride 16, 3.3M parameters total. Selected for CPU forward-pass
cost, not accuracy. Note that in vision-only 3D perception a larger 2D backbone
often *overfits* rather than helping, so "bigger is better" does not hold here
in any case.

ImageNet initialisation is why the Dataset applies ImageNet normalisation; the
two are coupled and a different pretraining source would require changing both.
The weights are baked into the container image so inference needs no network,
verified with `docker run --network none`.

**No training loop.**
Measured: 37.8 s per optimisation step at batch size 2 on CPU, so one epoch
over 324 training samples would take 102 minutes. Ten epochs from random 3D
heads would most likely converge to predicting `free` everywhere, since free is
the largest supervised class, and produce a near-zero mIoU. The compute buys
nothing the brief asks for.

Instead the pipeline is verified by fitting a single batch for 20 steps:
gradient reaches every one of the model's parameter tensors, and the loss falls
from 2.99 to 1.18. Starting loss agrees with ln(18) = 2.890, which is where an
untrained 18-class classifier should sit. This demonstrates that the backbone,
view transform, voxel encoder, head, loss and optimiser are all connected and
that the mask does not zero the signal — a stronger claim than a partially
trained model would support, at 13 minutes instead of hours.

**Consequences.**
- 98.3% of voxels project into at least one camera, so the backbone receives
  gradient from nearly the whole grid. The remaining 1.7% are supervised by the
  loss but contribute nothing to backbone learning — a structural property of
  backward projection, since gradient reaches the backbone only through voxels
  that land in-frame.
- Forward pass: 4.4 s at batch 1, 9.1 s at batch 2. Roughly linear, so
  per-sample cost is dominated by the view transform's per-camera loop rather
  than by the backbone.
- Train/validation is split by **scene**, not by sample: 324 train across 8
  scenes, 80 validation across 2. Keyframes are 0.5 s apart and highly
  correlated, so a sample-level split would place near-duplicate frames on both
  sides and report a meaningless validation score. The split is deterministic
  (last two scenes by name) rather than chosen.
- `data.num_frames` must be 1; the constructor raises otherwise. The temporal
  axis is present in the batch and the ego pose is stored in every shard, so
  temporal fusion is a fusion module rather than a refactor.