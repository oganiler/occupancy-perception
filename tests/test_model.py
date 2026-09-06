import torch

from occupancy.gt.grid import GridSpec
from occupancy.labels import NUM_CLASSES
from occupancy.models.backbone import ImageBackbone
from occupancy.models.loss import MaskedCrossEntropy
from occupancy.models.occupancy_net import OccupancyNet, voxel_centers
from occupancy.models.view_transform import NEAR_PLANE_M, NaiveProjection

SPEC = GridSpec((-4.0, -4.0, -1.0, 4.0, 4.0, 1.0), (1.0, 1.0, 1.0))  # 8x8x2


def _batch(B=1, N=2, H=64, W=64):
    K = torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]])
    # Camera 10 m behind the grid along -z, so the whole grid is in front of
    # it and within the field of view. Identity extrinsics put the camera at
    # the ego origin, which cannot see a grid centred on itself.
    T = torch.eye(4)
    T[2, 3] = -10.0
    return {
        "imgs": torch.randn(B, 1, N, 3, H, W),
        "intrinsics": K.expand(B, 1, N, 3, 3).contiguous(),
        "T_ego_from_cam": T.expand(B, 1, N, 4, 4).contiguous(),
        "target": torch.randint(0, NUM_CLASSES, (B, 8, 8, 2)),
    }

def _model():
    bb = ImageBackbone("resnet18", pretrained=False, out_stride=32)
    return OccupancyNet(bb, NaiveProjection((64, 64)), SPEC, hidden_channels=8)


def test_forward_shape():
    logits = _model()(_batch())
    assert logits.shape == (1, NUM_CLASSES, 8, 8, 2)


def test_voxel_centers_are_cell_centred():
    c = voxel_centers(SPEC)
    assert c.shape == (128, 3)
    assert torch.allclose(c[0], torch.tensor([-3.5, -3.5, -0.5]))


def test_voxels_behind_the_camera_are_rejected():
    """Negative depth projects to valid-looking pixels; it must be rejected.

    Two identity-extrinsic cameras see only the +z half-space. Voxels at
    negative z must therefore receive no features at all.
    """
    vt = NaiveProjection((64, 64))
    feats = torch.ones(1, 1, 4, 8, 8)
    K = torch.tensor([[[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]]]).unsqueeze(0)
    T = torch.eye(4).view(1, 1, 4, 4)

    centers = torch.tensor([[0.0, 0.0, 5.0],            # in front
                            [0.0, 0.0, -5.0],           # behind
                            [0.0, 0.0, NEAR_PLANE_M / 2]])  # inside near plane
    out = vt(feats, K, T, centers)
    assert out[0, :, 0].abs().sum() > 0
    assert out[0, :, 1].abs().sum() == 0
    assert out[0, :, 2].abs().sum() == 0


def test_ignored_voxels_contribute_no_gradient():
    """An all-ignored target must produce no gradient anywhere."""
    logits = torch.randn(1, NUM_CLASSES, 8, 8, 2, requires_grad=True)
    target = torch.full((1, 8, 8, 2), 255)
    loss, _ = MaskedCrossEntropy(ignore_index=255)(logits, target)
    assert torch.isnan(loss) or loss.item() == 0.0


def test_gradient_reaches_every_component():
    model, batch = _model(), _batch()
    loss, _ = MaskedCrossEntropy(ignore_index=255)(model(batch), batch["target"])
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"no gradient reached: {dead[:5]}"

def test_projection_hits_at_least_one_voxel():
    """Guards the fixture: a geometry where nothing projects in-frame would
    make the gradient test pass vacuously by never sampling anything."""
    vt = NaiveProjection((64, 64))
    batch = _batch()
    feats = torch.ones(1, 2, 4, 2, 2)
    out = vt(feats, batch["intrinsics"][:, 0],
             batch["T_ego_from_cam"][:, 0], voxel_centers(SPEC))
    assert (out.abs().sum(dim=1) > 0).any(), "no voxel projected into any view"    