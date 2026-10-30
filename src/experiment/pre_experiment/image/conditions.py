"""Image実験の全条件を保持し、有効条件だけを切り替える。"""

from dataclasses import dataclass
from typing import Literal, Mapping


ConditionId = Literal[
    "DP_BINOCULAR",
    "DP_MONO_DOM",
    "DP_MONO_NONDOM",
    "SP_NO_DEFOCUS",
    "SP_ALPHA_BLEND",
    "SP_DEFOCUS_SIMPLE",
    "SP_REPRO_LEFT_EYE",
    "SP_REPRO_RIGHT_EYE",
    "SP_MODEL_MEAN",
    "SP_MODEL_WTA",
    "SP_MODEL_GAIN",
]
EyeName = Literal["Left", "Right"]
ReproducedEye = Literal["", "Both", "Left", "Right"]


@dataclass(frozen=True)
class ImageConditionSpec:
    condition_id: ConditionId
    label: str
    plane_mode: Literal["dual", "single"]
    viewing_mode: Literal["binocular", "monocular"]
    reference_eye: Literal["dominant", "non_dominant", "both", "left", "right"]
    defocus_mode: Literal["physical", "none", "matched_simulation"]
    disparity_mode: Literal["physical", "none", "simple"]
    reproduced_eye: ReproducedEye = ""
    # 左右眼像の合成モデル：mean=1/2平均、wta=視認性WTA、gain=正規化ゲイン。
    combination_model: Literal["", "mean", "wta", "gain"] = ""


# 条件定義はすべて残す。実行対象はENABLED_CONDITION_IDSだけで切り替える。
ALL_CONDITION_SPECS: tuple[ImageConditionSpec, ...] = (
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
        "single", "binocular", "both", "matched_simulation", "simple", "Both",
    ),
    ImageConditionSpec(
        "SP_REPRO_LEFT_EYE",
        "Single Plane / reproduce left-eye view (monocular)",
        "single", "monocular", "left", "matched_simulation", "simple", "Left",
    ),
    ImageConditionSpec(
        "SP_REPRO_RIGHT_EYE",
        "Single Plane / reproduce right-eye view (monocular)",
        "single", "monocular", "right", "matched_simulation", "simple", "Right",
    ),
)
ALL_CONDITION_SPECS += (
    ImageConditionSpec("SP_ALPHA_BLEND", "Single Plane / alpha blend",
                       "single", "binocular", "both", "none", "none"),
    ImageConditionSpec("SP_MODEL_MEAN", "Single Plane / model1 mean",
                       "single", "binocular", "both", "matched_simulation", "simple", "Both", "mean"),
    ImageConditionSpec("SP_MODEL_WTA", "Single Plane / model2 WTA",
                       "single", "binocular", "both", "matched_simulation", "simple", "Both", "wta"),
    ImageConditionSpec("SP_MODEL_GAIN", "Single Plane / model3 normalized gain",
                       "single", "binocular", "both", "matched_simulation", "simple", "Both", "gain"),
)
CONDITION_BY_ID = {spec.condition_id: spec for spec in ALL_CONDITION_SPECS}

# ===== 条件スイッチ：通常はこのTrue / Falseだけを編集する =====
# ファイル保存後、実験を再起動する。起動済みセッションの設定は変更しない。
# matching：DP3条件＋SP_ALPHA_BLEND、10組×反復数（既定1）、計40×n試行。
# SP_DEFOCUS_SIMPLEは、defocus＋単純視差で作った左右眼像を融合して両眼観察。
# SP_REPRO_LEFT/RIGHT_EYEは対応する左眼/右眼だけで単眼観察する。
CONDITION_ENABLED: dict[ConditionId, bool] = {
    "DP_BINOCULAR":       True,   # DP両眼
    "DP_MONO_DOM":        True,   # DP利き眼のみ
    "DP_MONO_NONDOM":     True,   # DP非利き眼のみ
    "SP_ALPHA_BLEND":     True,  # matchingのみ：α値を調整
    "SP_NO_DEFOCUS":      False,  # SP：defocusなし
    "SP_DEFOCUS_SIMPLE":  False,   # SP：defocus＋左右眼像融合
    "SP_REPRO_LEFT_EYE":  False, # SP：左眼像を再現、左眼のみで観察
    "SP_REPRO_RIGHT_EYE": False, # SP：右眼像を再現、右眼のみで観察
    "SP_MODEL_MEAN":      True,  # SP：model1 左右眼像1/2平均（線形）
    "SP_MODEL_WTA":       True,  # SP：model2 視認性WTA
    "SP_MODEL_GAIN":      True,  # SP：model3 正規化ゲイン
}


def get_enabled_condition_ids(
    switches: Mapping[str, bool] | None = None,
) -> tuple[ConditionId, ...]:
    """スイッチを検証し、条件定義順で有効IDを返す。T/F文字列や0/1は不可。"""
    flags = CONDITION_ENABLED if switches is None else switches
    defined = set(CONDITION_BY_ID)
    if len(defined) != len(ALL_CONDITION_SPECS):
        raise ValueError("Image condition definitions contain duplicate IDs")
    missing = defined - set(flags)
    unknown = set(flags) - defined
    if missing or unknown:
        raise ValueError(
            f"CONDITION_ENABLED has missing IDs: {sorted(missing)}, "
            f"unknown IDs: {sorted(unknown)}"
        )
    invalid = [key for key, value in flags.items() if type(value) is not bool]
    if invalid:
        raise ValueError(
            "CONDITION_ENABLED values must be True or False: "
            + ", ".join(sorted(invalid))
        )
    enabled = tuple(
        spec.condition_id for spec in ALL_CONDITION_SPECS
        if flags[spec.condition_id]
    )
    if not enabled:
        raise ValueError("Enable at least one image condition in CONDITION_ENABLED")
    return enabled


# ===== 以下は自動生成。通常は編集しない =====
ENABLED_CONDITION_IDS: tuple[ConditionId, ...] = get_enabled_condition_ids()
DEFAULT_CONDITION_IDS = ENABLED_CONDITION_IDS
CONDITION_SPECS: tuple[ImageConditionSpec, ...] = tuple(
    CONDITION_BY_ID[condition_id] for condition_id in ENABLED_CONDITION_IDS
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
    if spec.reference_eye == "left":
        return "Left"
    if spec.reference_eye == "right":
        return "Right"
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