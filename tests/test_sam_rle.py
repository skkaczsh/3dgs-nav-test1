import numpy as np

from scripts.sam_rle import decode_rle, encode_uncompressed_rle
from semantic_eval.run_eval import decode_sam_segmentation


def encode_coco_counts(counts: list[int]) -> str:
    chars: list[str] = []
    for index, raw_value in enumerate(counts):
        value = raw_value - (counts[index - 2] if index > 2 else 0)
        while True:
            chunk = value & 0x1F
            value >>= 5
            more = value != (-1 if chunk & 0x10 else 0)
            chars.append(chr((chunk | (0x20 if more else 0)) + 48))
            if not more:
                break
    return "".join(chars)


def test_coco_compressed_rle_round_trips_column_major_mask() -> None:
    mask = np.asarray([[False, True, True], [True, False, False], [False, True, False]], dtype=bool)
    legacy = encode_uncompressed_rle(mask)
    compressed = {"size": legacy["size"], "counts": encode_coco_counts(legacy["counts"]), "encoding": "coco_rle"}

    assert np.array_equal(decode_rle(compressed), mask)
    assert np.array_equal(decode_sam_segmentation(compressed), mask)


def test_legacy_counts_remain_compatible() -> None:
    mask = np.asarray([[False, True], [True, False]], dtype=bool)
    assert np.array_equal(decode_rle(encode_uncompressed_rle(mask)), mask)
