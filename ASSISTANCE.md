# LLM Assistance Disclosure

The assignment asks that LLM-assisted work be distinguished from work I
reasoned about and implemented myself. This document is that record.

## Threshold applied

I count as **assisted** any code or prose whose first draft originated from an
LLM prompt. I do not count IDE line-completion, literature search, or reading
papers and documentation.

Everything marked *reasoned by me* is something I can explain live without
notes. That is the test I applied when writing this document, and where I could
not meet it, the row says so.

The work was carried out in an extended session with Claude (Anthropic), used
as a pair-programming and design-review partner rather than as a code
generator: I directed the work, chose between options, and reviewed every
output.

## Summary by area

### Environment and infrastructure

| Area | Reasoned by | Implemented by | Notes |
|---|---|---|---|
| Project decomposition and 4-day schedule | Assisted | — | Task breakdown, deliverable weighting, ordering |
| Repository scaffolding, LF enforcement, ignore rules | Assisted | Assisted | See "Errors found in assisted output" |
| Docker layer ordering, CPU-only torch | Me | Assisted | Rationale (build-cache behaviour, image size, no training required) is mine; Dockerfile drafted with assistance |
| Dependency version resolution | Assisted | Assisted | Version compatibility, including the `numpy<2` ceiling that nuscenes-devkit imposes, checked against PyPI metadata via LLM tooling |
| Dev container configuration | Assisted | Assisted | Kept in `.devcontainer/` so the deliverable compose file is untouched |
| Typed configuration layer | Assisted | Assisted | Structured-schema approach and the no-constructor-defaults rule were proposed in review; I adopted them after they caught two real bugs |

### Ground truth generation

| Area | Reasoned by | Implemented by | Notes |
|---|---|---|---|
| nuScenes adapter and SE(3) transform chain | Me | Assisted | Frames and the transform composition are familiar ground from my own work; code drafted with assistance and verified against the ground plane |
| **D-001** keyframe-only sweep aggregation | **Me** | Assisted | Using non-keyframes would make it impossible to distinguish model error from label error when evaluating. I want to see behaviour with real labels first. No KNN or spatial index is then needed in the GT path |
| Aggregation window size | **Me** | Assisted | Configurable; ±4 keyframes (±2 s), clamped at scene boundaries |
| **D-002** adoption of the Occ3D 18-class set | **Me** | — | Chose to match the published class set rather than define my own, so results remain comparable. I anticipated it would require only a converter between lidarseg and Occ3D categories, which proved correct |
| **D-002** deriving the mapping from the devkit | Assisted | Assisted | That `nuscenes.eval.lidarseg.utils.LidarsegClassMapper` collapses the 32 fine lidarseg classes onto indices 0..16, and that these coincide index-for-index with Occ3D 0..16, was found by the LLM inspecting the installed devkit source. `labels.py` was drafted with assistance. The decision to derive the table rather than transcribe it is mine |
| Voxelizer two-pass structure | **Me** | Assisted | Two passes so occupied-beats-free is structural rather than order-dependent (D-003). Reasoning is mine; code drafted with assistance |
| **D-004** Amanatides-Woo traversal | Cited | Assisted | Published prior work (see references). The `_dda` implementation was drafted with assistance, then reviewed in detail — I can explain the `t_max`/`t_delta` setup, why the traversal runs in voxel coordinates, and why termination at t > 1.0 needs no epsilon |
| **D-003** ray policy for corrected dynamic points | Assisted | Me | The conflict between a corrected endpoint and its original ray was raised in review; I chose Option D and the reference-sweep exception |
| **D-001 refinement** non-keyframe free-space pass | **Me** | **Me** | See "Contributions" below |
| **D-005** shard format | **Me** | Assisted | Deferred the format choice until the storage-layout measurement was available; chose the single-array sentinel so the stored array *is* the cross-entropy target |
| **D-006** `free_ray_stride` rejected | **Me** | Assisted | Measurement designed jointly; the interpretation that ray-level striding perforates free space into higher-entropy speckle, and that this is why files grow, was worked out in discussion |
| Voxelizer tests | Assisted | Assisted | Cases proposed and drafted with assistance; the properties they pin (precedence, determinism, the pass-3 guard) follow from decisions that are mine |

### Deep learning pipeline

| Area | Reasoned by | Implemented by | Notes |
|---|---|---|---|
| **D-007** aspect-preserving image resize | **Me** | Assisted | Options and their intrinsic-correction cost laid out with assistance; the choice, and the reasoning that reduced silent-failure surface outweighs pixel efficiency when no training occurs, is mine. I can explain why resizing without adjusting K produces a silently wrong projection rather than an error |
| Dataset, collate, scene-level split | Assisted | Assisted | The temporal-correlation leakage argument for splitting by scene rather than by sample was raised in review and I adopted it |
| **D-008** loss masking policy | Assisted | Assisted | The SparseOcc visible-mask ablation and its consequences were surfaced by the LLM with sources; the decision to leave the camera mask unimplemented but configurable is mine |
| **D-009** placeholder model architecture | Assisted | Assisted | The backward-vs-forward projection trade-off was explained in review; I chose the well-posed direction. Backbone, encoder and head are conventional choices drafted with assistance |
| Model and projection tests | Assisted | Assisted | Including the gradient-flow check, which caught a fixture geometry in which no voxel projected in-frame |

### Analysis and documentation

| Area | Reasoned by | Implemented by | Notes |
|---|---|---|---|
| Diagnostic scripts and coverage analysis | Assisted | Assisted | I directed which questions to investigate; scripts drafted with assistance |
| Documentation (this file, decisions, report, README) | Assisted | Assisted | Drafted with assistance from decisions I made and measurements I ran |

## Contributions that are mine

**The non-keyframe free-space pass (D-001 refinement).** My observation that
free space requires no semantic label, and that non-keyframe sweeps are
therefore usable for ray traversal even though nuScenes-lidarseg annotates
keyframes only.

I also identified that the existing two-pass ordering already made endpoint
handling unnecessary. The LLM had proposed three schemes for marking or
suppressing non-keyframe endpoints — mark occupied without a label, use a
fourth sentinel, terminate one voxel short — all of which the pass structure
makes moot: if the endpoint is never cast, there is nothing to handle.

The `UNOBSERVED`-only guard, which prevents free space being carved through the
interior of moving objects, was suggested in review. I then observed that the
guard could be unconditional rather than flag-controlled, since during the
free-space passes a voxel can only be `UNOBSERVED` or `FREE`, and writing
`FREE` to a `FREE` voxel is idempotent. That removed a parameter and made the
invariant structural.

**Design questions that redirected the work.** Asking whether the model input
should be matched to the LiDAR frame nearest the camera timestamp led to the
explicit treatment of why `t_ref` is the correct dynamic-correction target, and
to the residual limitation that the six cameras are not simultaneous with each
other.

## Errors found in assisted output

Recorded because they evidence that the output was reviewed rather than
accepted. Each was found by me and corrected.

| Error | Consequence if unnoticed |
|---|---|
| `.gitignore` pattern `data/` unanchored | Silently excluded `occupancy/data/` — the adapter would never have reached the repository, and a clean clone would fail on import |
| `free_sweep_stride` not added to the adapter constructor | Pass 3 was a no-op; the first run returned byte-identical results |
| `num_keyframes` used before it existed on `SampleData` | Shard provenance would have been missing |
| Configuration file written but never read | Every value came from constructor defaults; the config was dead code |
| Redundant `only_unobserved` flag in `_dda` | An extra parameter a caller could pass wrongly |
| Duplicate print left after an edit to a diagnostic | Cosmetic |

## Corrections to assisted analysis

Several explanations offered in review turned out to be wrong and were
overturned by measurement. The measurements, not the initial explanations, are
what the report states.

- **Ground-plane expectation stated in the wrong frame.** The road was
  predicted at z = -1.8 m (LiDAR frame) when the data is in the ego frame,
  where it belongs at 0. The measured 0.00 was initially described as evidence
  the calibration had been *skipped*, when it is evidence it was applied.
- **Voxelizer runtime overestimated by ~20x** (predicted "tens of seconds",
  measured 1.54 s/sample). This removed the performance argument for a native
  C++ backend.
- **Coverage variance attributed to scene enclosure.** That hypothesis predicts
  open intersections should score well; the two worst-covered scenes are both
  described as "wait at intersection". Replaced by the parallax explanation
  (r = -0.673 against ego displacement, n = 404).
- **Predicted that stationary scenes would gain most from the non-keyframe
  pass.** They gain five times less (1.2 points vs 6.0) at three times the
  cost, because non-keyframe sweeps from a stationary ego originate at the same
  position and strike the same beam radii.
- **`docker system prune -a` recommended** for a clean-build check, which would
  have discarded cached base layers and dependency downloads for no benefit.

## Open-source libraries and prior work

| Source | Used for | Reference |
|---|---|---|
| nuscenes-devkit 1.2.0 | Dataset access, sweep linking, transform chain, lidarseg-to-coarse class mapping | https://github.com/nutonomy/nuscenes-devkit |
| Amanatides & Woo (1987) | Ray traversal in the free-space passes. Chosen over fixed-step sampling (approximate — no step size is both complete and non-redundant) and over Bresenham (one voxel per major-axis step, so it omits voxels the ray crosses) | *A Fast Voxel Traversal Algorithm for Ray Tracing*, Eurographics '87, pp. 3-10. DOI 10.2312/egtp.19871000 |
| Occ3D-nuScenes | Grid specification (range, voxel size), 18-class label set, occupied/free/unobserved labelling rule | https://arxiv.org/abs/2304.14365 |
| SparseOcc | RayIoU metric; the visible-mask ablation quoted in D-008 | https://arxiv.org/abs/2312.17118 |
| BEVFormer / Lift-Splat-Shoot | Named alternative view transforms; the backward/forward projection trade-off in D-009 | https://arxiv.org/abs/2203.17270 |
| PyTorch, torchvision | Model, ImageNet-pretrained ResNet | https://pytorch.org |
| OmegaConf | Typed structured configuration | https://omegaconf.readthedocs.io |
