# lab
大石研究室における修士研究のためのリポジトリ

ost-ar-visibility-experiment/
├── README.md               ← プロジェクト概要・セットアップ手順
├── .gitignore              ← Git管理から除外するファイルの指定
├── .gitattributes          ← Git LFS の設定（大容量ファイル用）
├── requirements.txt        ← Python依存ライブラリ一覧
│
├── src/                    ← ソースコード
│   ├── stimulus/           ← 刺激生成スクリプト
│   │   ├── gabor_patch.py
│   │   └── noise_generator.py
│   ├── calibration/        ← キャリブレーション関連
│   │   ├── luminance_calibration.py
│   │   └── color_correction.py
│   ├── analysis/           ← 分析スクリプト
│   │   ├── response_analysis.py
│   │   └── visibility_model.py
│   └── utils/              ← ユーティリティ
│       └── image_utils.py
│
├── config/                 ← 実験設定ファイル
│   ├── experiment_config.yaml
│   └── display_params.json
│
├── data/                   ← 実験データ（Git LFS or .gitignore）
│   ├── raw/                ← 生データ（絶対に変更しない）
│   └── processed/          ← 処理済みデータ
│
├── results/                ← 解析結果・図表
│   ├── figures/
│   └── tables/
│
├── docs/                   ← ドキュメント・プロトコル
│   ├── experiment_protocol.md
│   └── calibration_log.md
│
├── notebooks/              ← Jupyter Notebook（探索的分析用）
│   └── exploratory_analysis.ipynb
│
└── tex/                    ← LaTeX論文原稿
    ├── main.tex
    └── references.bib