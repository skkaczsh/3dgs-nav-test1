from argparse import Namespace

from scripts.apply_semantic_geometry_guard import choose_label


def args(**overrides):
    base = dict(
        teacher_confidence_keep=0.65,
        floor_min_voxels=1200,
        floor_min_extent=2.5,
        floor_max_z_extent=0.9,
        wall_min_voxels=400,
        wall_max_normal_abs_z=0.62,
        surface_label_policy="veto_contradiction",
        demote_small_surfaces=False,
        allow_wall_to_floor=False,
        car_min_voxels=120,
        car_surface_normal_abs_z=0.88,
        car_surface_max_z_extent=0.35,
        railing_surface_normal_abs_z=0.88,
    )
    base.update(overrides)
    return Namespace(**base)


def test_surface_preserve_policy_keeps_floor_wall_labels():
    wall_on_mixed_patch = {
        "semantic_label": "wall",
        "geometry_type": "horizontal",
        "voxel_count": 100,
        "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 0.1]},
        "mean_normal": [0, 0, 1],
    }

    assert choose_label(wall_on_mixed_patch, args())[0] == "unknown"
    assert choose_label(wall_on_mixed_patch, args(surface_label_policy="preserve")) == (
        "wall",
        "kept_wall_surface_policy_preserve",
    )
