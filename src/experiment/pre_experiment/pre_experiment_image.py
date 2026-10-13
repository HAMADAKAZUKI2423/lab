# cd src && py -m experiment.pre_experiment.pre_experiment_image
"""conditions.pyのCONDITION_ENABLEDで指定した条件の画像予備実験を起動する。

各条件20ペア。今回の既定値はDP両眼・DP利き眼・DP非利き眼・
SP両眼（defocus＋左右眼像融合）の4条件、計80試行。
切替はimage/conditions.pyのTrue / Falseを編集し、保存後に再起動する。
"""

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
