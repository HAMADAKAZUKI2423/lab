# 保存先: lab/src/experiment/pre_experiment/pre_experiment_image.py
# 設定コードは image/config.py に保存する。この起動コードとは別ファイル。
# cd src && py -m experiment.pre_experiment.pre_experiment_image
"""conditions.pyのCONDITION_ENABLEDで指定した条件の画像予備実験を起動する。

各条件20ペア。今回の既定値はDP両眼・DP利き眼・DP非利き眼・
SP両眼（defocus＋左右眼像融合）の4条件、計80試行。
切替はimage/conditions.pyのTrue / Falseを編集し、保存後に再起動する。
既定は視角7.5°・上下±3.75°・連続提示の画像マッチング。旧評定は --task-mode rating。
matchingは全試行完了時だけ一括保存し、中断時の途中結果は保存しない。
実校正は必須。画面外は警告、SP色域外はクリッピングで継続する。
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
    parser.add_argument("--task-mode", choices=("matching", "rating"), default=None,
                        help="省略時はimage/config.pyのIMAGE_TASK_MODE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[image] 起動ファイル: {Path(__file__).resolve()}", flush=True)
    config = create_image_config(args.selected_pairs_dir, args.seed, args.task_mode)
    print(f"[image] 実験方式: {config.task_mode}", flush=True)
    app_class = ImageExperimentApp
    if config.task_mode == "matching":
        from .image.matching_app import ImageMatchingApp
        app_class = ImageMatchingApp
    root = tk.Tk()
    try:
        print("[image] 実験を準備します。画像フォルダ選択が出たら指定してください。", flush=True)
        app_class(root, config)
        print("[image] 表示・参加者入力を開始します。", flush=True)
        root.mainloop()
    finally:
        # Ctrl+C/例外でも途中結果を保存せず、GUIを閉じる。
        try:
            root.destroy()
        except tk.TclError:
            pass

if __name__ == "__main__":
    main()
