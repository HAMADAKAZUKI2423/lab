"""旧コマンド名を維持し、画像matching用の同じ3図を出力する。
cd src && py -m experiment.pre_analyze.analyze_image_dp_eye_difference [session_dir]
SP眼像再現のrating比較ではなく、DP単眼の実際の左右眼ゲインを比較する。
実装はpre_analyze_image.pyへ集約。両ファイルを同じフォルダへ保存する。
"""
if __package__:
    from .pre_analyze_image import main
else:
    from pre_analyze_image import main

if __name__ == '__main__':
    main()
