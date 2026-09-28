# cd src && py -m experiment.pre_experiment.pre_experiment_image
"""6条件×20画像ペアのImage evaluation予備実験を起動する。"""

import argparse
from pathlib import Path
import tkinter as tk

from .image import ImageExperimentApp, create_image_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-pairs-dir",
        type=Path,
        help="high10.csvとlow10.csvを含むselector結果フォルダ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="条件順・ペア順の基準seed（参加者IDと組み合わせて使用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    ImageExperimentApp(
        root,
        create_image_config(args.selected_pairs_dir, args.seed),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
