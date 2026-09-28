"""Image実験で使用する暫定5条件を宣言的に定義する。"""

from dataclasses import dataclass
from typing import Literal


ConditionId = Literal[
    "DP_BINOCULAR",
    "DP_MONO_DOM",
    "DP_MONO_NONDOM",
    "SP_NO_DEFOCUS",
    "SP_DEFOCUS_SIMPLE",
]
EyeName = Literal["Left", "Right"]


@dataclass(frozen=True)
class ImageConditionSpec:
    condition_id: ConditionId
    label: str
    plane_mode: Literal["dual", "single"]
    viewing_mode: Literal["binocular", "monocular"]
    reference_eye: Literal["dominant", "non_dominant", "both"]
    defocus_mode: Literal["physical", "none", "matched_simulation"]
    disparity_mode: Literal["physical", "none", "simple"]


CONDITION_SPECS: tuple[ImageConditionSpec, ...] = (
    ImageConditionSpec(
        "DP_BINOCULAR",
        "Dual Plane / binocular",
        "dual", "binocular", "both", "physical", "physical",
    ),
    ImageConditionSpec(
        "DP_MONO_DOM",
        "Dual Plane / dominant eye",
        "dual", "monocular", "dominant", "physical", "physical",
    ),
    ImageConditionSpec(
        "DP_MONO_NONDOM",
        "Dual Plane / non-dominant eye",
        "dual", "monocular", "non_dominant", "physical", "physical",
    ),
    ImageConditionSpec(
        "SP_NO_DEFOCUS",
        "Single Plane / no defocus",
        "single", "binocular", "dominant", "none", "none",
    ),
    ImageConditionSpec(
        "SP_DEFOCUS_SIMPLE",
        "Single Plane / defocus + simple disparity",
        "single", "binocular", "both", "matched_simulation", "simple",
    ),
)
CONDITION_BY_ID = {spec.condition_id: spec for spec in CONDITION_SPECS}
DEFAULT_CONDITION_IDS: tuple[ConditionId, ...] = tuple(
    spec.condition_id for spec in CONDITION_SPECS
)


def normalize_eye(value: str) -> EyeName:
    eye = str(value).strip().capitalize()
    if eye not in {"Left", "Right"}:
        raise ValueError(f"Eye must be Left or Right: {value!r}")
    return eye  # type: ignore[return-value]


def opposite_eye(eye: str) -> EyeName:
    return "Left" if normalize_eye(eye) == "Right" else "Right"


def resolve_reference_eye(spec: ImageConditionSpec, dominant_eye: str) -> str:
    dominant = normalize_eye(dominant_eye)
    if spec.reference_eye == "dominant":
        return dominant
    if spec.reference_eye == "non_dominant":
        return opposite_eye(dominant)
    return "Both"


def resolve_observed_eye(spec: ImageConditionSpec, dominant_eye: str) -> str:
    if spec.viewing_mode == "binocular":
        return "Both"
    return resolve_reference_eye(spec, dominant_eye)


def get_condition(condition_id: str) -> ImageConditionSpec:
    try:
        return CONDITION_BY_ID[condition_id]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unknown image condition: {condition_id}") from exc