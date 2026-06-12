import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_sam_mask_dirs_for_test",
        SCRIPTS / "compare_sam_mask_dirs.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mask_array_decodes_uncompressed_rle_fortran_order():
    module = load_module()
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    mask[2, 1] = True
    mask[1, 3] = True

    flat = mask.T.reshape(-1)
    counts = []
    previous = False
    run = 0
    first = True
    for value in flat:
        value = bool(value)
        if first:
            if value:
                counts.append(0)
                previous = True
            else:
                previous = False
            run = 1
            first = False
        elif value == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = value
    counts.append(run)

    decoded = module.mask_array({"segmentation": {"size": [3, 4], "counts": counts}})
    assert np.array_equal(decoded, mask)


def test_mask_array_keeps_binary_mask_schema():
    module = load_module()
    decoded = module.mask_array({"segmentation": [[True, False], [False, True]]})
    assert decoded.dtype == bool
    assert decoded.tolist() == [[True, False], [False, True]]
