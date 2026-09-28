"""Fukiage selectorが出力したhigh 10組・low 10組を検証して読み込む。"""

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class SelectedPair:
    pair_id: str
    difference_group: str
    group_rank: int
    global_rank: int
    foreground_texture_class: int
    foreground_path: Path
    background_path: Path
    pair_dir: Path
    right_visibility_score: float | None
    left_visibility_score: float | None
    absolute_score_difference: float | None


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if value == "" else float(value)


def _resolve_original(
    selection_dir: Path,
    pair_dir: Path,
    row: dict[str, str],
    role: str,
) -> Path:
    saved = (
        sorted(pair_dir.glob(f"{role}_original.*"))
        if pair_dir.is_dir() else []
    )
    if len(saved) > 1:
        raise ValueError(f"Multiple saved {role} images: {saved}")
    if saved:
        return saved[0].resolve()
    raw = Path(row[f"{role}_path"]).expanduser()
    path = raw if raw.is_absolute() else selection_dir / raw
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()



def load_selected_pairs(selection_dir: Path) -> list[SelectedPair]:
    selection_dir = selection_dir.expanduser().resolve()
    if not selection_dir.is_dir():
        raise FileNotFoundError(selection_dir)

    required = {
        "rank",
        "group_rank",
        "difference_group",
        "foreground_texture_class",
        "foreground_path",
        "background_path",
    }
    pairs: list[SelectedPair] = []
    for expected_group in ("high", "low"):
        csv_path = selection_dir / f"{expected_group}10.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{csv_path.name}: missing columns {missing}")
            rows = list(reader)
        if len(rows) != 10:
            raise ValueError(f"{csv_path.name} must contain 10 rows: {len(rows)}")

        for row in rows:
            group = row["difference_group"].strip().lower()
            if group != expected_group:
                raise ValueError(
                    f"{csv_path.name}: difference_group must be {expected_group}"
                )
            group_rank = int(row["group_rank"])
            pair_dir = selection_dir / f"{group}10" / f"rank_{group_rank:02d}"
            pair = SelectedPair(
                pair_id=f"{group}_{group_rank:02d}",
                difference_group=group,
                group_rank=group_rank,
                global_rank=int(row["rank"]),
                foreground_texture_class=int(row["foreground_texture_class"]),
                foreground_path=_resolve_original(
                    selection_dir, pair_dir, row, "foreground"
                ),
                background_path=_resolve_original(
                    selection_dir, pair_dir, row, "background"
                ),
                pair_dir=pair_dir.resolve(),
                right_visibility_score=_optional_float(
                    row, "right_visibility_score"
                ),
                left_visibility_score=_optional_float(
                    row, "left_visibility_score"
                ),
                absolute_score_difference=_optional_float(
                    row, "absolute_score_difference"
                ),
            )
            pairs.append(pair)

    pairs.sort(key=lambda item: (
        0 if item.difference_group == "high" else 1,
        item.group_rank,
    ))
    _validate_selected_pairs(pairs)
    return pairs


def _validate_selected_pairs(pairs: list[SelectedPair]) -> None:
    if len(pairs) != 20:
        raise ValueError(f"Exactly 20 selected pairs are required: {len(pairs)}")
    if len({pair.pair_id for pair in pairs}) != 20:
        raise ValueError("Pair_ID values are not unique")

    for group in ("high", "low"):
        group_pairs = [pair for pair in pairs if pair.difference_group == group]
        if {pair.group_rank for pair in group_pairs} != set(range(1, 11)):
            raise ValueError(f"{group}: group_rank must be exactly 1..10")
        if {pair.foreground_texture_class for pair in group_pairs} != set(range(1, 11)):
            raise ValueError(
                f"{group}: foreground_texture_class must be exactly 1..10"
            )

    foregrounds = {str(pair.foreground_path).casefold() for pair in pairs}
    backgrounds = {str(pair.background_path).casefold() for pair in pairs}
    if len(foregrounds) != 20:
        raise ValueError("Foreground images must be unique across the 20 pairs")
    if len(backgrounds) != 20:
        raise ValueError("Background images must be unique across the 20 pairs")
    if foregrounds & backgrounds:
        raise ValueError("The same source image is used as foreground and background")