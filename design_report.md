# Design Report

Occupancy-aware perception pipeline on nuScenes: ground-truth generation and a
training-ready deep learning pipeline.

---

## 1. The organising idea

Ground-truth generation and inference are not two halves of the same process.
They are opposites, and every structural decision follows from that.

| | GT generation | Inference |
|---|---|---|
| When | offline, once | online, every frame |
| Causality | **non-causal** — reads future frames | causal |
| Inputs | LiDAR, human-annotated 3D boxes | six camera images |
| Cost | seconds per sample | must fit a frame budget |
| Output | one voxel grid | one voxel grid |

The GT generator is allowed to cheat. It aggregates nine keyframes spanning
±2 s, uses annotations to correct moving objects, and produces a grid describing
one instant. The model gets that single instant's images and must reproduce the
answer. **That gap is the learning problem.**

This is physically true of the files: a shard contains one grid and six image
references. The nine keyframes that built it leave no trace beyond a
`num_keyframes` field recording how it was made.

---

## 2. Ground truth generation

### 2.1 Why aggregation is unavoidable

A single 32-beam sweep returns roughly 35,000 points. The grid is
200 × 200 × 16 = **640,000 voxels**. One sweep can touch at most ~5% of it, and
that is an upper bound before duplicates and out-of-range returns. Supervising
against that is hopeless.

### 2.2 Keyframe-only aggregation (D-001)

nuScenes-lidarseg annotates **keyframes only**: 404 lidarseg records against
31,206 `sample_data` records. Non-keyframe sweeps have geometry but no labels.

Two options:

- **A.** Aggregate keyframes only. Every point carries a measured label.
- **B.** Aggregate all 20 Hz sweeps and propagate labels by k-nearest-neighbour,
  as Occ3D does. Roughly 10× the points, but ~89% of them would carry inferred
  rather than measured labels.

**A was chosen.** Ground truth is the property the entire pipeline is
supervised against; Option B would place a heuristic inside it, making model
error indistinguishable from label error at evaluation time. The trade is point
density for label fidelity.

### 2.3 The transform chain

Points from each keyframe are expressed in the reference keyframe's ego frame:

```
p_ego_ref = T(ego_ref <- global) · T(global <- ego_i) · T(ego_i <- lidar_i) · p_lidar_i
```

Verified rather than assumed: `driveable_surface` points have a median z of
**0.00 m** with a ±0.45 m spread. The LIDAR_TOP extrinsic places the sensor
1.84 m above the ego origin, so a road at 0 in the ego frame is exactly what a
correctly applied calibration produces. Independently, Occ3D's z range of
[−1.0, 5.4] m only makes sense with the road near zero.

### 2.4 Dynamic objects

Aggregation assumes a static world. A vehicle at 10 m/s smears **40 m** across
the grid over a ±2 s window.

Points inside an annotated box are taken into that object's box-local frame at
time *i* and placed at the object's box pose at t_ref. Points whose instance is
not annotated at t_ref are dropped: we know they moved, but not to where.

t_ref is the correct target because a nuScenes sample is a hardware-synchronised
snapshot — camera exposure is triggered as the top LiDAR sweeps across each
camera's field of view. Correcting to any other time in the window would place
moving objects systematically ahead of or behind their appearance in the input
images, making the target unlearnable.

*Residual limitation:* the six cameras fire in sequence across one LiDAR
rotation (bounded by 50 ms at 20 Hz), so a single grid cannot be exactly
simultaneous with all six views. At 10 m/s that is roughly one voxel of
displacement. Not modelled; noted.

### 2.5 The three states, and the ray policy (D-003, D-004)

Occupied comes from ray endpoints. **Free space can only come from traversal** —
a voxel is free because a beam passed through it. Everything else is
`unobserved`.

Traversal uses Amanatides-Woo, which returns exactly the voxels a ray
intersects, in order, with no gaps or duplicates, at two floating-point
comparisons and one addition per step. Fixed-step sampling is approximate (no
step size is both complete and non-redundant); Bresenham takes one voxel per
major-axis step and therefore omits voxels the ray genuinely crosses. Both
would leave holes in free space that a BEV render would not reveal.

The traversal runs in continuous **voxel coordinates**, which sets the voxel
size to 1 and makes the ray parameter run 0 → exactly 1.0, so termination is an
exact comparison needing no epsilon.

Dynamic correction breaks the ray. A beam measured origin → surface at position
A; after correction the point is at B. For an approaching vehicle B lies
*between* origin and A, so the measured ray passes through the corrected object.
The policy: corrected points from non-reference keyframes mark occupied but cast
no free ray. The reference sweep is exempt, since there the correction is the
identity.

### 2.6 Pass structure

```
pass 1   free space from keyframe rays          -> FREE
pass 2   keyframe endpoints and semantic votes  -> OCCUPIED
pass 3   free space from non-keyframe rays      -> FREE   [optional]
```

Two passes rather than one is not stylistic. In a single-pass implementation,
whether a voxel ends up free or occupied would depend on the order points
appear in the array. Separating the passes makes **occupied beats free** a
property of the program's shape rather than of a conditional someone could
remove — which is also the precondition for a bit-exact parity test against any
future native backend.

`_dda` writes `FREE` only into `UNOBSERVED` voxels, so free-space passes are
additive and order-independent by construction. Only pass 2 overwrites.

Determinism is pinned by test.

### 2.7 Semantics (D-002)

lidarseg has 32 fine classes; Occ3D uses 18 (0–16 semantic, 17 free). The
devkit's `LidarsegClassMapper` already collapses 32 → 0..16, and those indices
coincide slot-for-slot with Occ3D's. The table is therefore **derived, not
transcribed** — a transcribed table would be correct the day it was written and
unverifiable thereafter.

Class 17 (`free`) is unreachable from any point label, asserted in code.

Verified visually: the BEV render of scene-0757 shows `driveable_surface` as the
largest class (1,005 columns) forming a connected road, bordered by `sidewalk`
and `terrain`, with `manmade` on the facades, and small objects (`pedestrian`,
`traffic_cone`) surviving voxelization at 0.4 m.

### 2.8 Storage (D-005)

One `uint8` array: 0–16 semantic, 17 free, **255 unobserved**. 255 is also
`loss.ignore_index`, so the stored array *is* the cross-entropy target — no
separate masking step to write or get wrong. The LiDAR visibility mask is
implicit (`target != 255`) rather than a second array that could drift.

Compression was measured, not assumed: 13× at 0.01 s per shard. Storing the
array transposed so z varies slowest improves this to 16×, worth ~5 MB across
the split — rejected, because a permanent axis-order asymmetry between memory
and disk is one forgotten transpose away from a silently rotated grid.

---

## 3. What the measurements showed

Full mini split, 404 samples: **4.06 s/sample**, 19.4 MB, unobserved 61.5%
(35.3–87.7%), free 35.6%, occupied 3.0%. About **38.6% of voxels carry
supervision**.

### Coverage is bounded by ego parallax

Not by scene enclosure, which was the initial hypothesis. Ego displacement
across the aggregation window correlates with unobserved fraction at
**r = −0.673** (n = 404).

| Scene | displacement | unobserved |
|---|---|---|
| scene-0553 | **0.0 m** | 81.1% |
| scene-0757 | 4.9 m | 80.8% |
| scene-1077 | 47.5 m | 52.3% |
| scene-0916 | 17.2 m | 47.9% |

The mechanism: a 32-beam LiDAR strikes the ground at 32 fixed radii determined
by beam elevation and sensor height, and that ring pattern is rigidly attached
to the sensor. Only **translation** moves where the rings land. Nine keyframes
from a stationary vehicle are nine identical clouds hitting the same 32 radii,
so the inter-beam gaps are geometrically unreachable.

The signature also appears in the variance: every scene varies 10–35 points
across its samples except scene-0553, which varies by **one** point (81–82%). A
stationary ego sees the same scene from the same place at every timestep.

The enclosure hypothesis predicted open intersections should score well. The two
worst-covered scenes are both described as "wait at intersection". It was
discarded.

Keyframe count matters far less: the full ramp from 5 to 9 aggregated keyframes
is worth 7.1 points, against 33 points across scene means.

### The non-keyframe free-space pass, and its boundary (D-001 refinement)

Free space needs no semantic label, so non-keyframe sweeps are usable for
traversal even though lidarseg annotates keyframes only. Implemented as pass 3.

| Scene | motion | unobserved, off → on | cost |
|---|---|---|---|
| scene-0061 | 17.7 m | 49.9% → 43.9% (**6.0 pts**) | 3.8 → ~11 s |
| scene-0553 | 0.0 m | 81.2% → 80.0% (**1.2 pts**) | 7.7 → 22.5 s |

The expectation was that stationary scenes would gain most. They gain **five
times less at three times the cost**. Non-keyframe sweeps from a stationary ego
originate at the same position, so the same beams strike the same radii. Pass 3
supplies additional *timesteps*, which become additional *viewpoints* only in
proportion to ego motion — it multiplies existing parallax but cannot create it.

Cost is roughly constant while benefit scales with motion, so the return is
worst exactly where coverage is worst. **Shipped disabled** (`free_sweep_stride:
0`): a full regeneration costs ~2 hours for ~4 points averaged across the split.
Enabling it is a one-line configuration change.

The guard was verified at scale: the occupied fraction is identical across all
strides in both scenes, including min and max, confirming pass 3 never
overwrites an occupied voxel across ~12.8 million voxels.

---

## 4. The DL pipeline

### 4.1 Modules and how they interact

```
shard + JPEGs
      │  Dataset            resize, scale K, normalise, add T axis
      ▼
imgs (B,T,6,3,288,512) + intrinsics + extrinsics
      │  backbone           shared weights; camera axis folded into batch
      ▼
feats (B,6,C,18,32)
      │  view transform     voxel centres project into each view
      ▼                     the camera axis collapses here
voxels (B,C,200,200,16)
      │  voxel encoder + head
      ▼
logits (B,18,200,200,16)  ──►  masked cross-entropy  ◄──  target (B,200,200,16)
```

All four model stages are injected; replacing any one touches only
`factory.py`.

### 4.2 The six images are fused by geometry, not concatenation

Stacking six images as 18 channels would be wrong: pixel (100, 100) in
`CAM_FRONT` and in `CAM_BACK` are unrelated, and channel stacking assumes a
spatial correspondence that does not exist.

Instead, one backbone processes all six independently, and fusion happens in 3D:
each voxel centre is projected into every camera, features are sampled where it
lands, and averaged over the cameras that saw it. **98.3% of voxels project into
at least one camera**, so the surround configuration covers the grid almost
completely.

### 4.3 Backward projection (D-009)

Forward projection (LSS) takes a pixel and must decide *where* in 3D to place
it — requiring a predicted depth distribution, which is the ill-posed part of
monocular vision.

Backward projection takes a voxel and finds its pixel. **No depth prediction is
needed**, because the 3D location is chosen rather than inferred: the grid is
the hypothesis space, and "if something were here, which pixel would show it?"
has one exact answer given K and the extrinsics.

The cost is ray ambiguity — voxels at 10 m and 30 m along the same ray sample
the same pixel and receive identical features. The network must disambiguate
from context. This is the documented weakness of the backward family, and why
FB-OCC combines both directions.

Two failure modes are guarded explicitly. A voxel *behind* a camera projects to
valid-looking pixels because the perspective divide cancels the sign, so `z > 0`
is required or the grid picks up mirrored content. And gradient reaches the
backbone only through voxels that land in-frame — a test with an unphysical
camera placement produced zero gradient throughout the backbone, correctly.

### 4.4 Image resize and the intrinsics (D-007)

Images are resized 1600 × 900 → 512 × 288, a scale of exactly 0.32 preserving
the native 16:9 aspect with no rounding, and both dimensions divisible by 32.

The BEVDet convention is crop-and-resize to 256 × 704, which spends its pixel
budget better by discarding sky that cannot contain an occupied voxel given the
grid's z range. It was rejected here because it requires **two** intrinsic
corrections — focal length and principal point scale with the resize, and the
principal point shifts again by the crop offset — against **one** for a pure
scale.

A wrong intrinsic produces no error: arrays stay the right size while the view
transform samples the wrong pixels. Halving the number of independent
corrections halves the opportunities for that. The trade favours the crop when
training, which is not in scope.

A unit test asserts that a 3D point lands at the same *normalised* image
position before and after resize, with a negative control confirming the test is
not vacuous.

### 4.5 Loss and masking (D-008)

Cross-entropy with `ignore_index=255`. Unobserved voxels produce no gradient —
the forward pass still emits a class there, but nothing corrects it. Unobserved
is a mask state, not a class: the head has 18 logits and none means "unknown".

The **camera** visibility mask is deliberately left unimplemented but
configurable. It is not a free improvement: SparseOcc report that training with
the visible mask yields 15 points *higher* voxel mIoU but 1 point *lower*
RayIoU, because the region behind each surface goes unsupervised and the model
fills it with duplicated predictions, producing a thicker, closer surface. Mask
on gives visible-surface reconstruction; mask off gives scene completion. For a
ground robot, conservative estimation of occluded volume plausibly matters more
than voxel mIoU. The question is left open rather than settled by default.

### 4.6 Scope: verified wiring, not training

Measured cost of training: **37.8 s per step** at batch size 2 on CPU, so one
epoch over 324 samples takes **102 minutes**. Ten epochs from random 3D heads
would most likely converge to predicting `free` everywhere.

Instead: fit one batch for 20 steps. **Gradient reaches every parameter tensor**
and loss falls **2.99 → 1.18**, with the initial value agreeing with
ln(18) = 2.890. That demonstrates backbone, view transform, voxel encoder, head,
loss and optimiser are all connected and the mask does not zero the signal — a
stronger claim than a partially trained model would support, in 13 minutes
rather than hours.

Train/validation is split **by scene** (324/80 across 8/2 scenes), never by
sample: keyframes are 0.5 s apart and highly correlated, so a sample-level split
would place near-duplicates on both sides and report a meaningless score.

---

## 5. Trade-offs and assumptions

| Decision | Chosen | Given up |
|---|---|---|
| Aggregation (D-001) | keyframes only | point density; ring gaps stay unobserved |
| Class set (D-002) | Occ3D 18, derived from devkit | finer classes a ground robot might want |
| Ray policy (D-003) | no free rays from corrected points | free-space coverage near moving objects |
| Storage (D-005) | natural memory order | 26% better compression |
| Ray striding (D-006) | disabled | 2× speed, at the cost of perforated free space |
| Non-keyframe pass | implemented, disabled | ~4 points coverage, for ~2 h regeneration |
| Resize (D-007) | aspect-preserving | 18% of pixels spent on sky |
| View transform (D-009) | backward projection | depth discrimination along rays |
| Training | wiring check only | nothing the brief asks for |

**Assumptions.** The ego is rigid and calibration is constant within a scene
(nuScenes records `calibrated_sensor` per scene, which the adapter honours).
Annotated boxes are accurate enough that box-local correction improves on no
correction. The Occ3D grid specification, designed for road vehicles, is a
reasonable starting point for ground robotics.

---

## 6. Extensions

**Highest value first.**

1. **Label propagation for stationary samples.** Coverage is parallax-bounded,
   and roughly 20% of the split is stationary. No amount of additional ray
   casting helps there — this is measured, not assumed. Only KNN propagation or
   mesh-based hole filling would.
2. **RayIoU.** Voxel mIoU cannot observe depth error and rewards thickened
   surfaces. RayIoU casts rays into predicted and ground-truth volumes and
   compares first-hit depth against thresholds. Its ray traversal is the same
   Amanatides-Woo kernel used in GT generation, so the algorithm is already
   present.
3. **A native voxelizer backend.** The `VoxelizerBackend` protocol and the
   determinism and precedence tests already form an executable specification for
   a bit-exact reimplementation. Deliberately not pursued: at 4.06 s/sample the
   performance argument does not hold, and traversal is ~90% of runtime so even
   an infinitely fast backend caps the end-to-end gain near 2×.
4. **LSS or deformable-attention view transform.** Both implement the existing
   `ViewTransform` protocol. LSS adds depth awareness at the cost of sparsity;
   deformable attention lets 3D queries attend to multi-view features.
5. **Temporal fusion.** The batch already carries a T axis and every shard
   stores the ego pose needed to warp between frames, so this is a fusion module
   rather than a refactor. A deployed system would more likely use recurrent BEV
   fusion — one backbone pass per frame regardless of context length — which
   requires state across calls that the stateless batch contract does not
   express, and introduces failure modes a stateless module does not have.
6. **Camera visibility mask**, to make the D-008 trade-off measurable rather
   than argued.

---

## 7. Notes on engineering approach

Three patterns recur, and each earned its place by catching a real error.

**Make invariants structural rather than conditional.** Two passes so precedence
cannot depend on ray order. `_dda` writing only into `UNOBSERVED` so free-space
passes are additive by construction, with no flag a caller could pass wrongly.
Grid dimensions derived from range and voxel size so the two cannot disagree.

**Defaults live in exactly one place.** The configuration schema. Components take
required constructor parameters, so none can be built in a state the
configuration does not describe. This caught `free_sweep_stride` never reaching
the adapter — which had already produced one silently no-op run.

**Static analysis surfaced under-specified interfaces, not typos.** Three times a
type error pointed at a contract weaker than the actual requirement: a union
assumed away, two array parameters that must travel together, and `nn.Module`
being too weak to promise `out_channels`. Each fix strengthened the interface;
a suppression would have hidden all three.

Measurements repeatedly overturned plausible explanations — voxelizer runtime by
20×, coverage variance attributed to the wrong cause, the non-keyframe pass
predicted to help most where it helped least. The numbers in this report are the
measurements, and where a prediction was wrong, the report says so.
