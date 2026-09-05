# tests/test_voxelizer.py
import numpy as np
import pytest

from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.gt.backends.reference import ReferenceVoxelizer
from occupancy.gt.grid import GridSpec
from occupancy.labels import FREE_CLASS


@pytest.fixture
def spec():
    # 10 x 10 x 10 of 1 m voxels, origin at (0, 0, 0)
    return GridSpec((0.0, 0.0, 0.0, 10.0, 10.0, 10.0), (1.0, 1.0, 1.0))


@pytest.fixture
def vox(spec):
    return ReferenceVoxelizer(spec, free_ray_stride=1)


def _one_ray(origin, point, label=4, cast_free=True):
    return (
        np.array([point], np.float32),
        np.array([label], np.uint8),
        np.array([origin], np.float32),
        np.array([cast_free], bool),
    )


def test_axis_aligned_ray_marks_the_path_free(vox):
    state, _ = vox.build(*_one_ray((0.5, 0.5, 0.5), (7.5, 0.5, 0.5)))
    assert (state[0:7, 0, 0] == FREE).all()
    assert state[7, 0, 0] == OCCUPIED
    assert (state[8:, 0, 0] == UNOBSERVED).all()   # nothing beyond the endpoint


def test_endpoint_beats_free_regardless_of_order(vox):
    """D-003 precedence: a return outranks a traversal."""
    pts = np.array([[7.5, 0.5, 0.5], [3.5, 0.5, 0.5]], np.float32)
    state, _ = vox.build(
        pts, np.array([4, 4], np.uint8),
        np.tile(np.array([[0.5, 0.5, 0.5]], np.float32), (2, 1)),
        np.array([True, True]),
    )
    # voxel 3 is both traversed by ray 0 and the endpoint of ray 1
    assert state[3, 0, 0] == OCCUPIED


def test_no_free_ray_still_marks_occupied(vox):
    """Corrected dynamic points mark occupied without carving free space."""
    state, _ = vox.build(*_one_ray((0.5, 0.5, 0.5), (7.5, 0.5, 0.5),
                                   cast_free=False))
    assert state[7, 0, 0] == OCCUPIED
    assert (state[0:7, 0, 0] == UNOBSERVED).all()


def test_out_of_bounds_points_are_cropped(vox):
    state, _ = vox.build(*_one_ray((0.5, 0.5, 0.5), (99.0, 99.0, 99.0)))
    assert state.shape == (10, 10, 10)          # no crash, no wraparound


def test_majority_vote(vox):
    pts = np.tile(np.array([[5.5, 5.5, 5.5]], np.float32), (3, 1))
    _, sem = vox.build(pts, np.array([4, 4, 10], np.uint8),
                       np.tile(np.array([[0.5, 0.5, 0.5]], np.float32), (3, 1)),
                       np.array([True, True, True]))
    assert sem[5, 5, 5] == 4


def test_free_voxels_carry_the_free_class(vox):
    state, sem = vox.build(*_one_ray((0.5, 0.5, 0.5), (7.5, 0.5, 0.5)))
    assert (sem[state == FREE] == FREE_CLASS).all()


def test_determinism(vox):
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (500, 3)).astype(np.float32)
    args = (pts, rng.integers(0, 17, 500).astype(np.uint8),
            np.zeros((500, 3), np.float32) + 0.5, np.ones(500, bool))
    a, b = vox.build(*args), vox.build(*args)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])