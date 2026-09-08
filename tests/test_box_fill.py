import numpy as np

from occupancy.data.nuscenes_adapter import Box3D
from occupancy.gt.backends.base import FREE, OCCUPIED, UNOBSERVED
from occupancy.gt.box_fill import apply_box_fill
from occupancy.gt.grid import GridSpec

SPEC = GridSpec((-4.0, -4.0, -1.0, 4.0, 4.0, 1.0), (1.0, 1.0, 1.0))   # 8x8x2


def _box(name="vehicle.car", half=(1.0, 1.0, 0.5)):
    T = np.eye(4)                              # centred at the ego origin
    return Box3D(T_ego_from_box=T, half_extent_m=np.array(half),
                 label=4, name=name)


def _grids(fill_value):
    return (np.full(SPEC.dims, fill_value, np.uint8),
            np.zeros(SPEC.dims, np.uint8))


def test_correct_turns_interior_free_into_unobserved():
    state, sem = _grids(FREE)
    n = apply_box_fill(state, sem, (_box(),), SPEC, "correct")
    assert n["corrected"] > 0
    assert (state == UNOBSERVED).sum() == n["corrected"]


def test_correct_never_touches_occupied():
    """A near-surface voxel marked by pass 2 must survive."""
    state, sem = _grids(FREE)
    state[4, 4, 1] = OCCUPIED
    apply_box_fill(state, sem, (_box(),), SPEC, "correct")
    assert state[4, 4, 1] == OCCUPIED


def test_correct_does_not_assert_occupancy():
    state, sem = _grids(UNOBSERVED)
    n = apply_box_fill(state, sem, (_box(),), SPEC, "correct")
    assert n["asserted"] == 0
    assert (state == OCCUPIED).sum() == 0


def test_assert_fills_unobserved_with_the_box_class():
    state, sem = _grids(UNOBSERVED)
    n = apply_box_fill(state, sem, (_box(),), SPEC, "assert")
    assert n["asserted"] > 0
    assert (sem[state == OCCUPIED] == 4).all()


def test_non_vehicle_boxes_are_ignored():
    state, sem = _grids(FREE)
    n = apply_box_fill(state, sem, (_box(name="human.pedestrian.adult"),),
                       SPEC, "correct")
    assert n["corrected"] == 0


def test_off_is_a_no_op():
    state, sem = _grids(FREE)
    before = state.copy()
    apply_box_fill(state, sem, (_box(),), SPEC, "none")
    assert np.array_equal(state, before)