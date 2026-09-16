"""Metadata inspection and ranked get-up motion search."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow.parquet as pq

from apex_getup.demonstrations.paths import BonesDatasetPaths


SEARCH_TERMS: dict[str, int] = {
    "lying to standing": 12,
    "floor to standing": 12,
    "stand up": 10,
    "get up": 10,
    "getting up": 9,
    "supine": 8,
    "prone": 8,
    "rise": 5,
    "rising": 5,
}

SEARCH_FIELDS = (
    "content_name",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
    "content_type_of_movement",
    "content_body_position",
)


@dataclass(frozen=True)
class MotionCandidate:
    score: int
    motion_id: str
    path: str
    frames: int
    description: str
    movement_type: str
    body_position: str
    package: str
    category: str
    is_mirror: bool
    props: str


def find_getup_candidates(
    paths: BonesDatasetPaths,
    *,
    include_mirrors: bool = False,
    limit: int | None = None,
) -> list[MotionCandidate]:
    columns = [
        "filename", "move_duration_frames", "move_g1_path", "package", "category",
        "is_mirror", "content_props", *SEARCH_FIELDS,
    ]
    rows = pq.read_table(paths.metadata, columns=columns).to_pylist()
    candidates: list[MotionCandidate] = []
    for row in rows:
        if row["is_mirror"] and not include_mirrors:
            continue
        filename_text = str(row["filename"] or "").lower().replace("_", " ")
        searchable = " | ".join(str(row[field] or "") for field in SEARCH_FIELDS).lower().replace("_", " ")
        score = sum(
            weight * (searchable.count(term) + 2 * filename_text.count(term))
            for term, weight in SEARCH_TERMS.items()
        )
        # Prop-assisted recoveries are valid candidates but less useful for a
        # self-contained get-up policy than floor-only motions.
        props = str(row["content_props"] or "0")
        if props not in ("0", "none", "None", ""):
            score -= 30
        if not score or not row["move_g1_path"]:
            continue
        candidates.append(
            MotionCandidate(
                score=score,
                motion_id=str(row["filename"]),
                path=str(row["move_g1_path"]),
                frames=int(row["move_duration_frames"]),
                description=str(row["content_short_description"] or row["content_natural_desc_1"] or ""),
                movement_type=str(row["content_type_of_movement"] or ""),
                body_position=str(row["content_body_position"] or ""),
                package=str(row["package"] or ""),
                category=str(row["category"] or ""),
                is_mirror=bool(row["is_mirror"]),
                props=props,
            )
        )
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.motion_id))
    return candidates if limit is None else candidates[:limit]
