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