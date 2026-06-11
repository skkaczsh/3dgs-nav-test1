#!/usr/bin/env python3
"""Shared scene-aware VLM prompts for dense semantic labeling.

The semantic route uses VLM output as structured observations for point-cloud
fusion. These prompts intentionally constrain the model to the rooftop scan
context and to the fixed taxonomy used by project_semantic.py.
"""

from __future__ import annotations

import json


ROOFTOP_SCENE_CONTEXT = """\
Scene:
- The data is an incremental MANIFOLD/Mid360 rooftop scan with synchronized cameras.
- A high ratio of large horizontal roof/ground surface is expected and is not an error.
- Sky and distant background must be ignored for point-cloud semantics.
- Thin metal structures, railings, pipes, equipment edges, and cables often touch or overlap floor-like roof surfaces in 2D masks.
- A common failure mode is labeling thin railings/equipment as floor because the mask includes nearby light brown/gray roof pixels.
"""


SEGMENTATION_GOAL = """\
Segmentation goal:
- Build dense point-level semantics, not an image caption.
- Treat floor/wall/building as large stable surface layers.
- Treat railing/pipe/equipment as fine foreground targets that should not be absorbed into floor just because the 2D mask touches roof pixels.
- Sky, distant background, invalid borders, and lens artifacts should not become valid point-cloud objects.
- If one highlighted mask contains both a large surface and a fine target, report the dominant physical target and mark the observation as mixed when the schema supports it.
"""


LABEL_TAXONOMY = {
    "unknown": {
        "id": 0,
        "parent_class": "unknown",
        "description": "Insufficient visual evidence or out-of-taxonomy content.",
    },
    "other": {
        "id": 1,
        "parent_class": "other",
        "description": "Valid foreground object that does not match a more specific label.",
    },
    "wall": {
        "id": 2,
        "parent_class": "surface",
        "description": "Vertical planar wall, parapet side, or retaining surface.",
    },
    "floor": {
        "id": 3,
        "parent_class": "surface",
        "description": "Large horizontal walkable roof/ground/platform surface.",
    },
    "ceiling": {
        "id": 4,
        "parent_class": "surface",
        "description": "Overhead planar surface; uncommon in this rooftop dataset.",
    },
    "grass": {
        "id": 5,
        "parent_class": "vegetation",
        "description": "Low vegetation or lawn.",
    },
    "tree": {
        "id": 6,
        "parent_class": "vegetation",
        "description": "Tree trunk, branches, or canopy.",
    },
    "person": {
        "id": 7,
        "parent_class": "dynamic",
        "description": "Human body or clothing.",
    },
    "car": {
        "id": 8,
        "parent_class": "dynamic",
        "description": "Vehicle.",
    },
    "railing": {
        "id": 9,
        "parent_class": "structure",
        "description": "Guardrail, fence, handrail, or thin continuous metal barrier.",
    },
    "building": {
        "id": 10,
        "parent_class": "structure",
        "description": "Building facade, roof structure, large architectural structure.",
    },
    "sky": {
        "id": 11,
        "parent_class": "background",
        "description": "Sky only. Must be ignored during point-cloud semantic projection.",
    },
    "road": {
        "id": 12,
        "parent_class": "surface",
        "description": "Outdoor road or paved ground outside the rooftop surface.",
    },
    "water": {
        "id": 13,
        "parent_class": "background",
        "description": "Water surface.",
    },
    "furniture": {
        "id": 14,
        "parent_class": "object",
        "description": "Movable furniture-like object.",
    },
    "pipe": {
        "id": 15,
        "parent_class": "structure",
        "description": "Pipe, cable tray, exposed conduit, or linear utility element.",
    },
    "equipment": {
        "id": 16,
        "parent_class": "object",
        "description": "Rooftop device, cabinet, box, HVAC-like unit, sensor, or fixture.",
    },
    "ignore": {
        "id": 255,
        "parent_class": "background",
        "description": "Do not project to valid point semantics.",
    },
}


MASK_LABEL_SCHEMA = {
    "label": "one taxonomy label",
    "parent_class": "taxonomy parent class",
    "confidence": 0.0,
    "description": "short free-text physical description, e.g. white HVAC outdoor unit",
    "identity_hint": "stable instance hint useful for object fusion, e.g. white rectangular HVAC unit near parapet",
    "attributes": {
        "color": "dominant visible color or empty string",
        "material": "visible material if obvious or empty string",
        "shape": "compact shape phrase if obvious or empty string",
        "function": "specific function/type if obvious, such as HVAC outdoor unit, pipe, guardrail",
    },
    "mixed": False,
    "is_large_surface": False,
    "can_merge_to_surface": False,
    "ambiguous_with": ["optional taxonomy labels"],
    "reason": "short visual reason",
}


MERGE_DECISION_SCHEMA = {
    "decision": "merge | keep_split | uncertain",
    "confidence": 0.0,
    "physical_relation": "same_object | adjacent_touching | overlapping_mask_only | different_objects | unclear",
    "reason": "short explanation",
    "evidence": ["brief visual evidence"],
    "risk": "main risk if this decision is wrong",
}


def taxonomy_lines() -> str:
    rows = []
    for label, meta in LABEL_TAXONOMY.items():
        rows.append(
            f"- {label}: id={meta['id']}, parent={meta['parent_class']}. {meta['description']}"
        )
    return "\n".join(rows)


def mask_label_prompt(extra_context: str | None = None) -> str:
    """Prompt for classifying one highlighted segmentation mask."""
    sections = [
        ROOFTOP_SCENE_CONTEXT,
        SEGMENTATION_GOAL,
        "Task:\nClassify only the highlighted mask for point-cloud semantic projection.",
        "Rules:\n"
        "- Return only strict JSON.\n"
        "- Use exactly one label from the taxonomy below.\n"
        "- Do not invent labels or synonyms.\n"
        "- Keep label coarse and fixed, but use description/identity_hint/attributes to describe the physical instance.\n"
        "- Examples: label=equipment with description='white HVAC outdoor unit'; label=pipe with description='thin gray conduit along wall'.\n"
        "- The highlighted mask/overlay may use artificial palette colors. Never copy overlay colors into description or attributes.\n"
        "- Describe color/material only from the underlying original RGB image; leave color/material empty when uncertain.\n"
        "- If the highlighted mask mixes a large surface and a thin object, set mixed=true.\n"
        "- If the mask is mostly sky or distant background, use sky or ignore.\n"
        "- Prefer railing/pipe/equipment for thin foreground structures even when they touch floor pixels.\n"
        "- Use unknown when the evidence is insufficient.",
        "Taxonomy:\n" + taxonomy_lines(),
        "Return schema:\n" + json.dumps(MASK_LABEL_SCHEMA, ensure_ascii=False),
    ]
    if extra_context:
        sections.insert(1, "Additional context:\n" + extra_context.strip())
    return "\n\n".join(sections)


def merge_review_prompt(item: dict) -> str:
    """Prompt for reviewing whether two object candidates should be merged."""
    proposal = item["proposal"]
    return f"""\
{ROOFTOP_SCENE_CONTEXT}

{SEGMENTATION_GOAL}

Task:
Review whether the two long-range objects in this contact sheet should be merged into one physical object.
The sheet contains representative segmentation overlays from object A and object B.
Labels like a0/a1 belong to object A; b0/b1 belong to object B.

Merge only if the visual evidence suggests the two sides are the same physical object or the same continuous structure.
Keep split if they are merely near each other, touch at an edge, are both on the same roof surface, or are caused by a coarse mask covering nearby but distinct structures.
Use uncertain if the image evidence is insufficient.

Candidate metadata:
- review_id: {item["review_id"]}
- object_a: {proposal["object_a"]}
- object_b: {proposal["object_b"]}
- candidate_a: {proposal["candidate_a"]}
- candidate_b: {proposal["candidate_b"]}
- score: {proposal["score"]}
- centroid_distance: {proposal.get("centroid_distance", "")}
- bbox_distance: {proposal.get("bbox_distance", "")}
- bbox_overlap_ratio: {proposal.get("bbox_overlap_ratio", "")}
- color_distance: {proposal.get("color_distance", "")}
- same_source_cluster: {proposal.get("same_source_cluster", "")}

Return only strict JSON with this schema:
{json.dumps(MERGE_DECISION_SCHEMA, ensure_ascii=False)}
"""


def main() -> None:
    print(mask_label_prompt())


if __name__ == "__main__":
    main()
