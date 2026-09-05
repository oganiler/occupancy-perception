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
  not a class, and is never predicted by the model. See D-00X (loss masking)
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

**Attribution.** The algorithm is published prior work, cited above and in
ASSISTANCE.md. The implementation in `occupancy/gt/backends/reference.py` is
this project's own.