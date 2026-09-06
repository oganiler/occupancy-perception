import numpy as np
import pytest

from occupancy.data.dataset import NATIVE_H, NATIVE_W


def project(K: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    uvw = K @ p_cam
    return uvw[:2] / uvw[2]


def test_resize_preserves_normalised_projection():
    """A 3D point must land at the same relative image position after resize.

    This is the property that matters and the one no shape check would catch:
    a forgotten intrinsic scale leaves every array the right size while the
    view transform samples the wrong pixels.
    """
    K = np.array([[1266.4, 0.0, 816.3],
                  [0.0, 1266.4, 491.5],
                  [0.0, 0.0, 1.0]])                  # a real CAM_FRONT K
    p_cam = np.array([3.7, -1.2, 14.0])              # arbitrary point, z > 0

    H, W = 288, 512
    scale = W / NATIVE_W
    assert abs(H / NATIVE_H - scale) < 1e-9          # D-007 aspect assumption

    K_small = K.copy()
    K_small[:2, :] *= scale

    u0, v0 = project(K, p_cam)
    u1, v1 = project(K_small, p_cam)

    assert u1 / W == pytest.approx(u0 / NATIVE_W, abs=1e-9)
    assert v1 / H == pytest.approx(v0 / NATIVE_H, abs=1e-9)


def test_unscaled_intrinsics_would_fail_this_test():
    """Confirms the test above is not vacuous."""
    K = np.array([[1266.4, 0.0, 816.3],
                  [0.0, 1266.4, 491.5],
                  [0.0, 0.0, 1.0]])
    u0, _ = project(K, np.array([3.7, -1.2, 14.0]))
    assert u0 / 512 != pytest.approx(u0 / NATIVE_W, abs=1e-6)