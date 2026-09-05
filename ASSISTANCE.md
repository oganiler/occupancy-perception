# LLM Assistance Disclosure

## Threshold used

## Summary

| Area | Reasoned by | Implemented by | Notes |
|---|---|---|---|
| Project decomposition and schedule | Assisted | — | 4-day breakdown developed in conversation with an LLM |
| Docker layer ordering, CPU-torch choice | Me | Assisted | Rationale (build-cache behaviour, image size, no training required) is mine; Dockerfile drafted with assistance |
| Dependency version resolution | Assisted | Assisted | Version compatibility checked against PyPI metadata via LLM tooling |
| Lidar Keyframe-only sweep aggregation | Me | Assisted | Using non-key frames will make it impossible to distinguish model error from label error when evaluating. I want to see performance with real labels first. No need KNN or spatial index needed in the GT path for labeling non-key frames|
|NuScenes Adapter Window Size| Me | Assisted | Configurable. Start with ±4 keyframes (±2 s), clamped at scene boundaries|
|NuScenes Adapter Class Remapping| Me | Assisted | lidarseg has 32 categories; Occ3D uses 18. match Occ3D's 18-class set exactly (better for comparing against published numbers). The devkit already produces Occ3D's indices. nuscenes.eval.lidarseg.utils.LidarsegClassMapper collapses the 32 fine lidarseg classes to 16 coarse ones, with an ignore class at index 0. Its get_coarse2idx() returns exactly: 1 barrier · 2 bicycle · 3 bus · 4 car · 5 construction_vehicle · 6 motorcycle · 7 pedestrian · 8 traffic_cone · 9 trailer · 10 truck · 11 driveable_surface · 12 other_flat · 13 sidewalk · 14 terrain · 15 manmade · 16 vegetation|

## Open-source libraries used

| Library | Purpose | Reference |
|---|---|---|
| nuscenes-devkit | Dataset access, sweep linking, transform chain | https://github.com/nutonomy/nuscenes-devkit |
