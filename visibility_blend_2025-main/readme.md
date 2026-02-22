# Visibility Blending
視認性予測モデルおよび視認性保存半透明レンダラー  
画像合成・ネットワーク学習およびその他注意点についての説明

## 環境構築
Python 実行環境を前提としています。以下は一例です。

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` には `ColorVideoVDP` を開発モードでインストールする指定（`-e ./ColorVideoVDP`）が含まれているため、リポジトリルートでコマンドを実行してください。

## コード説明
| ファイル名 | 用途 | 
| --- | --- | 
|`compute_vismap_single.py` | 単一画像ペアに対して視認性マップを計算する |
|`blending_images.py`     | 各手法による画像合成を行う |
|`blending_videos.py`     | 各手法による動画合成を行う |
|`calibrate_model_2024.py`| 視認性マッチングデータを用いた視認性予測モデルの校正を行う |
|`network_train.py`       | アルファマップ予測ネットワークの学習を行う |
|`utils.py`               | 刺激データ (`stimulus`) の構築・画像入出力など汎用処理 |

## compute_vismap_single.py
単一の背景・前景ペアに対して視認性マップを推定し、グレースケールおよびヒートマップ画像として保存する補助スクリプト。

### 典型的な使い方
```bash
python compute_vismap_single.py \
    --model vismlp_norm \
    --bg imgs/parasol.png \
    --fg imgs/boy.png \
    --alpha_map imgs/boy_vis_50_30.png \
    --out_gray vismaps/vismap_gray.png \
    --out_heat vismaps/vismap_heat.png \
    --out_blend vismaps/blend.png
```

- `--model` は `vismodel/vismodel_configs.json` に記載のモデル名を指定する。
- `--bg` と `--fg` は同じ解像度の画像を指定する。`--alpha`（スカラー）か `--alpha_map`（グレースケール画像）でアルファブレンド時の透過率を与える。
- `--out_gray` は必須。`--out_heat`、`--out_blend` は任意。`--cmap` でヒートマップ時のカラーマップ（例: `plasma`）を指定可能。
- `--device` で `cuda:0` のように GPU を明示できる。指定しない場合は CPU 実行。

補助オプションとして、既に合成済み画像を確認用に保存したい場合は `--blend` にパスを指定する。`--blend` を省略した場合のみ、`--alpha`/`--alpha_map` の情報から内部でブレンド画像を生成する。

## blending_images.py
各手法による画像合成を行う
```
python blending_images.py
```
| オプション | 用途 | 
| --- | --- | 
| `--device` | 使用GPU指定 | 
| `-b` `--blender` | 使用合成手法を記載したJSONファイルを指定<br>デフォルトは`default_blenders.json` | 
| `-i` `--image` | 入力画像を記載したJSONファイルを指定<br>デフォルトは`default_images.json`| 
| `-o` `--output` | 結果出力先を指定<br>デフォルトは`/results/blend_images/` | 
| `--not_check` | `--blender`で指定したファイルに記載の`check_vismodel`による、各手法の合成結果の視認性確認を行わない | 

その他オプションにより、`dataset`ディレクトリで定義された画像集を利用して画像合成を行う。

### `--blender`オプション記載方法
サンプルは`default_blenders.json`参照
| key | value type | 用途 | 
| --- | --- | --- | 
| `check_vismodel` | `str` | 合成結果の視認性を評価するvismodelを指定する |
| `model` | `list[dict]` | 合成手法を指定する |

#### `model`指定方法
| key | value type | 用途 | 
| --- | --- | --- | 
| `type` | `str` | 合成手法指定<br>指定名は`blender/loader.py`内`load_blender`関数の`type`を参照 |
| `shortname` | `str` | 保存時のディレクトリ/ファイル名に利用 |
| `target_type` | `content`or`background` | 合成画像の視認性保存対象を指定 |

視認性予測モデルでの再帰調整手法における損失設定等、その他のオプションについてもここで指定する。  
`blender/loader.py`内`load_blender`関数の`blender_dict`として読み込まれているため、確認されたい。

### `--image`オプション記載方法
サンプルは`default_images.json`参照  
コードは`utils.py`内`stimulus.set_from_imgdict`関数を参照  
`key:image, value:list[dict]`により指定する。以下、各`dict`でのitemについて説明する。
| key | value type | 用途 | 
| --- | --- | --- | 
| `bg` | `str` | 背景画像のパスを指定 |
| `opaque`,`content` | `str` | 前景画像を指定する<br>`opaque`は不透明画像,`content`は半透明画像を入力時に指定 |
| `mask` | `str` | マスク画像のパスを指定<br>指定しない場合は画像全体をマスク領域とする。<br>`content`での半透明画像入力時は、`alpha > 0`の領域をマスク領域とする。全体を指定したい場合は、`full`と入力する。|
| `vismap`, `vis_value` | `str`,`float` | 目標視認性マップを指定する<br>マップもしくは`float`による数値で一様に指定する |

## `network_train.py`使用方法
ネットワークの学習を行う
```
python network_train.py
```
| オプション | 用途 | 
| --- | --- | 
| `--device` | 使用GPU指定 | 
| `--setting` | 学習設定を記載したJSONファイルを指定<br>デフォルトは`default_train.json` | 
| `-o` `--output` | 結果出力先を指定<br>デフォルトは`/results/trains/` | 
| `--epochs` | 学習エポック数<br>デフォルトは500 | 
| `--save_data_interval` | ネットワークパラメーター保存周期<br>デフォルトは100 | 
| `--save_image_interval` | 学習時画像サンプル出力周期<br>デフォルトは25 | 
| `--batch_size` | バッチサイズ<br>デフォルトは48 | 
| `--lr` | 学習率<br>デフォルトは0.001 | 

### `--setting`オプション記載方法
サンプルは`default_train.json`参照  
`key:setting value:list[dict]`により指定する。  
以下、各`dict`でのitemについて説明する。
| key | value type | 用途 | 
| --- | --- | --- | 
| `network` | `dict` | 学習ネットワークのtypeを指定する<br>`networks/utils.py`内の`load_network`関数を参照 |
| `vismodel` | `str` | 学習に使用する視認性予測モデルの構成を指定する<br>`vismodel/vismodel_configs.json`および`vismodel/utils.py`内の`load_vismodel`関数を参照 |
| `shortname` | `str` | 保存時のディレクトリ/ファイル名に利用 |
| `target_type` | `content`or`background` | 学習時の視認性保存対象を指定 |
| `wo_semi` | `bool` | `True`時、データセットが不透明画像のみを出力する |
| `tv_type` | `map`or`scalar` | データセットの目標視認性の形式を指定<br>`network/models/INetwork`の`INetwork.tv_input`を参照 |
| `loss_type` | `str` | 損失の計算方式を指定<br>`vismodel/loss/loader`の`load_lossFunction`関数を参照 |

## ディレクトリ説明
| ディレクトリ名 | 用途 | 
| --- | --- | 
| `blender` | 各画像合成手法の実装<br>`readme`あり | 
| `dataset` | 各データセット実装および生データ <br>`readme`あり| 
| `imgs` | 合成用画像 | 
| `networks` | 画像合成用ネットワーク<br>`readme`あり | 
| `results` | 実験結果保存 | 
| `videos` | 合成用動画 | 
| `vismodel` | 視認性予測モデル実装<br>`readme`あり | 

## その他
視認性予測モデルの構成は`vismodel/vismodel_configs.json`に記載  
ネットワークの構成は`networks/network_configs.json`に記載