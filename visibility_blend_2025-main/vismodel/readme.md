# 視認性予測モデル実装
## 使用方法
`./supermodels/visModel.py`に親モデルを記載  

`VisModel.set_xxx`関数で入力及び前処理を行う。  
`VisModel.compute_visibility`関数で視認性予測を行う。  
`VisModel.compute_weights`関数で重み計算のみを行う。  
`VisModel.compute_visibility_wo_weight`関数で重み計算以降の視認性予測を行う。

`vismodel_configs.json`において、
小林知幾修論において使用したコンテンツ視認性予測モデルは`FeatureWeight_Solo`であり、
MLP視認性予測モデルは`MLP_3way_ds4_h2_c64_lp2_do_botelow_csig2_zero_wd1e4`である。


## 実装モデル
`./utils.py`内の`load_vismodel`関数を参照  
`vismodel_configs.json`に示す構成から、`load_vismodel`を用いて読み込まれる
| モデル/`type` | 説明 | 
| --- | --- | 
| `vismodel_describe.py` | コンテンツ適応視認性予測モデル`VisModel_Describe` | 
| `vismodel_mlp.py` | MLP視認性予測モデル`VisModel_MLP` |

## `VisModel_Describe`
| オプション | type | 説明 | 
| --- | --- | --- |
| `level` | int | ラプラシアンピラミッド層数 | 
| `weight_mode` | str | 重み計算方法設定 | 
| `residual_fit` | bool | ラプラシアンピラミッド残差成分に対して線形フィッティング処理を行うかどうか | 
| `extraction_mode` | `normal`or`partial` | 線形フィッティング方法<br>`normal`は通常線形フィッティング、`partial`は偏相関係数に基づく高精度線形フィッティング | 
| `sigmoid_param` | int | 一般化ロジスティック関数パラメーター<br>未設定の場合は`./eval/eval_recalc_userstudy_vis.py`によってフィッティングできる |

### `weight_mode`設定
| `weight_mode` | 説明 | 
| --- | --- | 
| `content_adaptive_multi` | 重み計算に、目標画像から参照画像を線形フィッティングによって取り除いたラプラシアンピラミッドを用いる | 
| `content_adaptive_solo` | 重み計算に、目標画像のラプラシアンピラミッドを用いる<br>ただし、残差成分は目標画像と参照画像の差分を用いる | 
| `static` | 画像によらない固定重み | 
| `spatial_adaptive_multi` | 各色・ラプラシアンレベルの重みを画像によらず等しくして、空間領域で重み付けを行う。<br>計算方法は`content_adaptive_multi`と同じ | 
| `spatial_weighted_adaptive_multi` | 各色・ラプラシアンレベルの重みを画像によらず等しくして、空間領域で重み付けを行う。<br>計算方法は`content_adaptive_solo`と同じ  | 
| `spatial_adaptive_multi` | 各色・ラプラシアンレベルの重みを画像によらず一定にして、空間領域で重み付けを行う。<br>計算方法は`spatial_weighted_adaptive_solo`と同じ | 
| `spatial_adaptive_solo` | 各色・ラプラシアンレベルの重みを画像によらず一定にして、空間領域で重み付けを行う。<br>計算方法は`content_adaptive_solo`と同じ  | 

## `VisModel_MLP`
| オプション | type | 説明 | 
| --- | --- | --- |
| `level`                   | int | ラプラシアンピラミッド層数 | 
| `corr_ksize`              | int | 線形フィッティングの窓サイズ | 
| `weight_mode`             | str | 計算方式設定 | 
| `extraction_mode`         |str | 線形フィッティング方法 | 
| `sigmoid_type`            |str | 一般化ロジスティック関数形式 | 
| `use_lowpass_diff_op_bl`  |bool | `weight_mode`が`2-way`のとき、目標画像の残差成分として、残差成分と合成画像の差分を用いる | 
| `num_hidden_layer`        |int | 隠れ層数 | 
| `mlp_dim`                 |int | 隠れ層チャンネル数 | 
| `skip_dn`                 |bool | `True`時、除算正規化を行わず、Normalizationを行う | 
| `norm_mode`               |`none`or`bn` | `skip_dn`が`True`時に使うNormalizationの種類 | 
| `no_mask`                 |bool | responseに対してアルファマスクを適応しない | 
| `nobound_opaque`          |bool | opaque target生成時に，アルファマスクを適用しない| 
| `fc_downsample_factor`    |int | 隠れ層のダウンサンプリング倍率 | 
| `drop_out_rate`           |float | ドロップアウト倍率 | 
| `lp_norm`                 |float | 視認性マップをスコアに変換する際のノルム | 
| `mask_loss_weight`        |float | `no_mask`が`True`時、アルファマスク外の視認性マップ値が0になるように学習させる| 
| `adaptive_max_vis`        |bool |`sigmoid_type`が`custom_sigmoid_v2`ではない時、視認性予測値の上限を学習可能にする | 

### `weight_mode`設定
| `weight_mode` | 説明 | 
| --- | --- | 
| `original` | コンテンツ適応視認性予測モデルと同じ動作 | 
| `3-way` | 合成画像、対象画像、参照画像の3つを結合したものをMLPに入力する | 
| `2-way` | 対象画像と、合成画像から参照画像を減算したものの2つを結合したものをMLPに入力する | 
| `2-way-extract` | 対象画像と、合成画像内の対象画像成分の2つを結合したものをMLPに入力する | 