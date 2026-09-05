import numpy as np

from occupancy.labels import (
    FREE_CLASS,
    NUM_CLASSES,
    OCC3D_CLASS_NAMES,
    build_fine_to_occ3d_lut,
    remap_labels,
)

# Pinned so a devkit upgrade cannot silently change the class mapping.
# Regenerate deliberately, never by copying a failing run's output.
EXPECTED_LUT = [
    0, 0, 7, 7, 7, 0, 7, 0, 0, 1, 0, 0, 8, 0, 2, 3, 3, 4, 5, 0, 0, 6, 9, 10, 11, 12, 13, 14, 15, 0, 16, 0
]


def test_class_set_shape():
    assert NUM_CLASSES == 18
    assert len(OCC3D_CLASS_NAMES) == NUM_CLASSES
    assert OCC3D_CLASS_NAMES[FREE_CLASS] == "free"
    assert OCC3D_CLASS_NAMES[0] == "others"


def test_lut_is_pinned(nusc):
    lut = build_fine_to_occ3d_lut(nusc)
    assert lut.dtype == np.uint8
    assert lut.tolist() == EXPECTED_LUT


def test_free_is_unreachable_from_a_point_label(nusc):
    """Occ3D class 17 is assigned by ray traversal, never by a point (D-002)."""
    assert build_fine_to_occ3d_lut(nusc).max() < FREE_CLASS


def test_remap_preserves_shape(nusc):
    lut = build_fine_to_occ3d_lut(nusc)
    raw = np.arange(32, dtype=np.uint8)
    out = remap_labels(raw, lut)
    assert out.shape == raw.shape
    assert out.dtype == np.uint8
    assert out.max() <= 16