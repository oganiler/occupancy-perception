"""Adapter isolating the rest of the pipeline from the nuScenes devkit.

Everything downstream depends on the dataclasses defined here, never on
`NuScenes`. Swapping datasets means writing a new adapter, not touching the
voxelizer or the DL pipeline.

Aggregation policy is keyframe-only (see docs/decisions.md D-001): every
returned point carries a measured lidarseg label. Points are expressed in the
ego frame of the *reference* keyframe, so the temporal window is invisible to
consumers -- they receive a single cloud at a single timestamp.
""" 
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix
from pyquaternion import Quaternion

from occupancy.labels import build_fine_to_occ3d_lut, remap_labels

LIDAR_CHANNEL = "LIDAR_TOP"
LIDAR_SWEEPS_PER_KEYFRAME = 10   # 20 Hz LiDAR against 2 Hz keyframes. For Non-Key Frames

@dataclass(frozen=True)
class AggregatedCloud:
    """Multi-keyframe point cloud, ego frame of the reference keyframe."""

    points_m: np.ndarray           # (N, 3) float32
    labels: np.ndarray             # (N,)   uint8, Occ3D classes 0..16
    sensor_origins_m: np.ndarray   # (N, 3) float32, ray origin per point
    in_annotated_box: np.ndarray         # (N,)   bool, point was inside an annotated box
    from_reference: np.ndarray     # (N,)   bool, point came from the reference sweep
    # Non-keyframe sweep rays: traversal only, endpoints never cast.
    # No labels, because a traversed voxel is class 17 regardless of what the
    # beam eventually struck.
    free_only_points_m: np.ndarray    # (M, 3) float32
    free_only_origins_m: np.ndarray   # (M, 3) float32    

    def __len__(self) -> int:
        return int(self.points_m.shape[0])


@dataclass(frozen=True)
class CameraData:
    name: str
    image_path: str
    intrinsic: np.ndarray          # (3, 3) float64
    T_ego_from_cam: np.ndarray     # (4, 4) float64


@dataclass(frozen=True)
class SampleData:
    """Everything one training sample needs, free of devkit types."""

    sample_token: str
    scene_name: str
    timestamp_us: int
    num_keyframes: int             # actually aggregated, after boundary clamping
    cloud: AggregatedCloud
    cameras: tuple[CameraData, ...]
    T_global_from_ego: np.ndarray  # (4, 4), reference keyframe ego pose


class NuScenesAdapter:
    """Reads nuScenes and emits SampleData.

    A single NuScenes instance is held for the adapter's lifetime; constructing
    one costs seconds on mini and minutes on trainval, so it must not be done
    per sample.
    """

    def __init__(
        self,
        dataroot: str,
        version: str,
        window_keyframes: int,
        min_distance_m: float,
        separate_dynamic: bool,
        camera_names: tuple[str, ...],
        free_sweep_stride: int,
    ) -> None:
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.window_keyframes = window_keyframes
        self.min_distance_m = min_distance_m
        self.separate_dynamic = separate_dynamic
        self.camera_names = camera_names
        self.free_sweep_stride = free_sweep_stride
        self._lut = build_fine_to_occ3d_lut(self.nusc)   # (32,) uint8, built once

    # -- public ------------------------------------------------------------

    def sample_tokens(self) -> list[str]:
        return [s["token"] for s in self.nusc.sample]

    def _sweep_to_ref(
        self, sd_token: str, T_egoref_from_global: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one LiDAR sweep into the reference ego frame.

        Returns (points (3, N) float64, sensor origin (3,) float64).
        Shared by keyframe and non-keyframe loading; only the label handling
        differs between them.
        """
        sd = self.nusc.get("sample_data", sd_token)
        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc.dataroot, sd["filename"])
        )
        pts = pc.points[:3, :]

        keep = np.linalg.norm(pts[:2, :], axis=0) >= self.min_distance_m
        pts = pts[:, keep]

        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        T_global_from_lidar = (
            transform_matrix(np.array(pose["translation"]),
                             Quaternion(pose["rotation"]))
            @ transform_matrix(np.array(cs["translation"]),
                               Quaternion(cs["rotation"]))
        )
        T = T_egoref_from_global @ T_global_from_lidar
        return _apply(T, pts), T[:3, 3]    

    def _nonkeyframe_rays(
        self, ref_sd_token: str, T_egoref_from_global: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Free-space rays from non-keyframe sweeps in the aggregation window.

        These sweeps carry no lidarseg labels, but free space needs none: a
        traversed voxel is class 17 whatever the beam struck. Their endpoints
        are never cast, so no unlabelled-occupied voxel is ever created.

        Dynamic objects are not corrected here -- annotations exist only on
        keyframes. The voxelizer's UNOBSERVED-only guard prevents these rays
        from contradicting the box-corrected occupancy pass 2 writes.
        """
        empty = np.zeros((0, 3), np.float32)
        if self.free_sweep_stride <= 0:
            return empty, empty

        span = self.window_keyframes * LIDAR_SWEEPS_PER_KEYFRAME
        tokens: list[str] = []
        for direction in ("prev", "next"):
            found: list[str] = []
            tok = self.nusc.get("sample_data", ref_sd_token)[direction]
            for _ in range(span):
                if not tok:
                    break
                sd = self.nusc.get("sample_data", tok)
                if not sd["is_key_frame"]:
                    found.append(tok)
                tok = sd[direction]
            # Stride within each direction so the sampling stays symmetric
            # about the reference keyframe.
            tokens.extend(found[:: self.free_sweep_stride])

        if not tokens:
            return empty, empty

        pts_all, org_all = [], []
        for tok in tokens:
            pts, origin = self._sweep_to_ref(tok, T_egoref_from_global)
            n = pts.shape[1]
            pts_all.append(pts.T)
            org_all.append(np.repeat(origin.reshape(1, 3), n, axis=0))

        return (
            np.concatenate(pts_all).astype(np.float32),
            np.concatenate(org_all).astype(np.float32),
        )    

    def get(self, sample_token: str) -> SampleData:
        ref_sample = self.nusc.get("sample", sample_token)
        ref_sd_token = ref_sample["data"][LIDAR_CHANNEL]
        ref_sd = self.nusc.get("sample_data", ref_sd_token)
        ref_pose = self.nusc.get("ego_pose", ref_sd["ego_pose_token"])

        T_global_from_egoref = transform_matrix(
            np.array(ref_pose["translation"]), Quaternion(ref_pose["rotation"])
        )
        T_egoref_from_global = transform_matrix(
            np.array(ref_pose["translation"]), Quaternion(ref_pose["rotation"]),
            inverse=True,
        )

        # Boxes at the reference time, keyed by instance, in global frame.
        ref_boxes = self._boxes_by_instance(ref_sd_token)

        window = self._keyframe_window(sample_token)
        chunks = [
            self._load_keyframe(tok, ref_sd_token, T_egoref_from_global, ref_boxes)
            for tok in window
        ] 

        free_pts, free_org = self._nonkeyframe_rays(
            ref_sd_token, T_egoref_from_global
        )

        cloud = AggregatedCloud(
            points_m=np.concatenate([c[0] for c in chunks]).astype(np.float32),
            labels=np.concatenate([c[1] for c in chunks]).astype(np.uint8),
            sensor_origins_m=np.concatenate([c[2] for c in chunks]).astype(np.float32),
            in_annotated_box=np.concatenate([c[3] for c in chunks]),
            from_reference=np.concatenate([c[4] for c in chunks]),
            free_only_points_m=free_pts,
            free_only_origins_m=free_org,
        )     

        scene = self.nusc.get("scene", ref_sample["scene_token"])
        return SampleData(
            sample_token=sample_token,
            scene_name=scene["name"],
            timestamp_us=ref_sample["timestamp"],
            num_keyframes=len(window),
            cloud=cloud,
            cameras=self._cameras(ref_sample),
            T_global_from_ego=T_global_from_egoref,
        )

    # -- window selection --------------------------------------------------

    def _keyframe_window(self, ref_token: str) -> list[str]:
        """Reference keyframe plus +/- window_keyframes, clamped at scene ends.

        Scene boundaries are a real case, not a hypothetical: with 40 keyframes
        per 20 s scene, 8 of every 40 samples sit within the window of an edge.
        """
        tokens = [ref_token]

        tok = ref_token
        for _ in range(self.window_keyframes):
            tok = self.nusc.get("sample", tok)["prev"]
            if not tok:
                break
            tokens.append(tok)

        tok = ref_token
        for _ in range(self.window_keyframes):
            tok = self.nusc.get("sample", tok)["next"]
            if not tok:
                break
            tokens.append(tok)

        return tokens

    # -- per-keyframe loading ---------------------------------------------

    def _load_keyframe(
        self,
        sample_token: str,
        ref_sd_token: str,
        T_egoref_from_global: np.ndarray,
        ref_boxes: dict[str, Box],
    ) -> tuple[np.ndarray, ...]:
        sample = self.nusc.get("sample", sample_token)
        sd_token = sample["data"][LIDAR_CHANNEL]
        sd = self.nusc.get("sample_data", sd_token)

        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc.dataroot, sd["filename"])
        )
        pts_lidar = pc.points[:3, :]                              # (3, N)

        raw = np.fromfile(
            os.path.join(
                self.nusc.dataroot,
                self.nusc.get("lidarseg", sd_token)["filename"],
            ),
            dtype=np.uint8,
        )                                                          # (N,)
        assert raw.shape[0] == pts_lidar.shape[1], "label/point count mismatch"

        # Drop ego-vehicle returns. Done manually rather than with
        # LidarPointCloud.remove_close, which would desync the label array.
        keep = np.linalg.norm(pts_lidar[:2, :], axis=0) >= self.min_distance_m
        pts_lidar = pts_lidar[:, keep]
        labels = remap_labels(raw[keep], self._lut)

        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        T_global_from_lidar = (
            transform_matrix(np.array(pose["translation"]), Quaternion(pose["rotation"]))
            @ transform_matrix(np.array(cs["translation"]), Quaternion(cs["rotation"]))
        )

        pts_global = _apply(T_global_from_lidar, pts_lidar)        # (3, N)

        # Sensor origin: the lidar's position, i.e. the transform's translation.
        origin_global = T_global_from_lidar[:3, 3]

        is_dyn = np.zeros(pts_global.shape[1], dtype=bool)
        if self.separate_dynamic:
            pts_global, is_dyn, labels = self._correct_dynamic(
                sd_token, pts_global, labels, ref_boxes
            )

        pts_ref = _apply(T_egoref_from_global, pts_global).T       # (N, 3)
        origin_ref = _apply(
            T_egoref_from_global, origin_global.reshape(3, 1)
        ).T                                                        # (1, 3)

        n = pts_ref.shape[0]
        return (
            pts_ref,
            labels,
            np.repeat(origin_ref, n, axis=0),
            is_dyn,
            np.full(n, sd_token == ref_sd_token, dtype=bool),
        )

    # -- dynamic object handling -------------------------------------------

    def _boxes_by_instance(self, sd_token: str) -> dict[str, Box]:
        out: dict[str, Box] = {}
        for box in self.nusc.get_boxes(sd_token):
            ann = self.nusc.get("sample_annotation", box.token)
            out[ann["instance_token"]] = box
        return out

    def _correct_dynamic(
        self,
        sd_token: str,
        pts_global: np.ndarray,      # (3, N)
        labels: np.ndarray,          # (N,)
        ref_boxes: dict[str, Box],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Move points on annotated objects to where that object is at t_ref.

        Without this, a car at 10 m/s smears 40 m across the grid over a
        +/-2 s window. Each object's points are taken into its own box-local
        frame at time i, then placed at that object's box pose at t_ref.

        Points belonging to an object that is not annotated at t_ref are
        dropped: we know they moved, but not to where.
        """
        is_dyn = np.zeros(pts_global.shape[1], dtype=bool)
        drop = np.zeros(pts_global.shape[1], dtype=bool)
        corrected = pts_global.copy()

        for box in self.nusc.get_boxes(sd_token):
            inside = points_in_box(box, pts_global)                # (N,) bool
            if not inside.any():
                continue

            is_dyn |= inside
            instance = self.nusc.get("sample_annotation", box.token)["instance_token"]
            ref_box = ref_boxes.get(instance)
            if ref_box is None:
                drop |= inside
                continue

            T_box_from_global = transform_matrix(
                box.center, box.orientation, inverse=True
            )
            T_global_from_refbox = transform_matrix(
                ref_box.center, ref_box.orientation
            )
            corrected[:, inside] = _apply(
                T_global_from_refbox @ T_box_from_global, pts_global[:, inside]
            )

        keep = ~drop
        return corrected[:, keep], is_dyn[keep], labels[keep]

    # -- cameras -----------------------------------------------------------

    def _cameras(self, sample: dict) -> tuple[CameraData, ...]:
        cams = []
        for name in self.camera_names:
            sd = self.nusc.get("sample_data", sample["data"][name])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            cams.append(
                CameraData(
                    name=name,
                    image_path=os.path.join(self.nusc.dataroot, sd["filename"]),
                    intrinsic=np.array(cs["camera_intrinsic"], dtype=np.float64),
                    T_ego_from_cam=transform_matrix(
                        np.array(cs["translation"]), Quaternion(cs["rotation"])
                    ),
                )
            )
        return tuple(cams)


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (3, N) points -> (3, N)."""
    return T[:3, :3] @ pts + T[:3, 3:4]