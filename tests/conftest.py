import os
import pytest

from occupancy.config import load_config
from occupancy.factory import build_adapter


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def nusc(cfg):
    if not os.path.isdir(os.path.join(cfg.dataset.root, cfg.dataset.version)):
        pytest.skip(f"nuScenes not found at {cfg.dataset.root}")
    return build_adapter(cfg).nusc