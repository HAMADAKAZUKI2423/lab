# Script Quick Start

## compute_vismap_single.py
- 単一画像ペアの視認性マップを計算する。
- 背景 (`--bg`)、前景 (`--fg`)、透過率 (`--alpha`) を指定し、出力をパスで指定する:

```
python compute_vismap_single.py \
    --model vismlp_norm \
    --bg imgs/parasol.png \
    --fg imgs/boy.png \
    --alpha 0.5 \
    --out_blend vismaps/blend.png \
    --out_gray vismaps/vismap_gray.png \
    --out_heat vismaps/vismap_heat.png
```

- `--model` には使用する視認性予測モデル名を指定する。出力パスは存在しない場合、自動でディレクトリが作成されないため事前に用意しておく。


## blending_images.py
- プロジェクトルートで実行する。
- 既定の合成設定と画像リストを読み込む例:

```
python blending_images.py -b settings_blender/settings_blenders_default.json -i default_images.json
```

- 結果は `results/blend_images/` 以下に保存される。必要に応じて `-o` で出力先を変更できる。

