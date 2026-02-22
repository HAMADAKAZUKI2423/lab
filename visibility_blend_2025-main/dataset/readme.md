# 使用データセット
ネットワーク学習用データセットと、画像合成用データセットを定義する  
`./dataset`ディレクトリ内に、ネットワーク学習用データセットの生データを配置する(Cocoデータ/Dtdデータ等)

## ネットワーク学習用データセット
| ファイル | 説明 | 
| --- | --- | 
| `train_coco_dataset.py` | Coco2017Valデータセットを前景・背景に用いたデータセット`CocoDataset`を定義 | 
| `train_coco_dtd_dataset.py` | Coco2017Valデータセットを前景に、Dtdデータセットを背景に用いたデータセット`CocoDtdDataset`を定義 | 

### `CocoDataset`オプション
| オプション | type | 説明 | 
| --- | --- | --- | 
| `target_type` | `content`or`background` | 視認性保存対象を指定<br>目標視認性値の出力範囲が異なる | 
| `device` | `torch.device` | 出力デバイスを指定 |
| `wo_semi` | `bool` | `True`時出力画像が不透明画像に設定 |
| `half_semi` | `bool` | `wo_semi`が`False`で、`half_semi`が`True`時、出力画像が半分の確率で不透明画像に設定 |
| `tv_type`| `map`or`scalar` |目標視認性マップの形式を指定 |
| `uniform_map`|`bool` |`tv_type`が`map`で、`uniform_map`が`True`時、出力目標視認性マップを一様に設定<br>`False`時(デフォルト)ではCocoのアノテーションに沿って2値に設定されている|

## 画像合成用データセット
`./IBlendDataset.py`にコントローラーを記載  
`../blending_images.py`のオプションとして指定して利用する  
前景・背景の組み合わせを複数作成し、画像合成を行う

### 種類
`./loader.py`を参照
| データ | 説明 | 
| --- | --- | 
| `exp` | `imgs/exp/target/opacity`内の不透明画像を`img/exp/background`内の背景に対して合成する | 
| `exptrans` | `imgs/exp/target/transparent`内の半透明画像を`img/exp/background`内の背景に対して合成する | 
| `exptranstgbg` | `imgs/exp/target/transparent`内の半透明画像を`img/exp/target/opacity`内の画像を背景として合成する| 
| `res` | `imgs/exp/hd`内の画像を、異なる解像度で合成する | 
| `test` | `imgs/exp/test`内の人工画像に対して合成を行う | 
| `expalpha` | `imgs/exp/target/opacity`内の不透明画像を、一定のアルファ値を持つ半透明画像として変換し、`img/exp/background`内の背景に対して合成する | 
| `cocodtd` | `CocoDtdDataset`内の画像に対して合成を行う | 
| `coco` | `CocoDataset`内の画像に対して合成を行う | 