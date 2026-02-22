# 画像合成実装
## 使用方法
`./models/IBlender.py`にコントローラーを記載  
`IBlender`を継承したクラスにて次の関数により画像合成が可能。
```
IBlender.blend(stimlus) # 画像合成
IBlender.save_imgs(save_path) # 合成画像保存
```
`stimlus`は`../utils.py`で定義された入力を指定するクラスを参照  
`./loader.py`にて各合成クラスを呼び出す

## 実装済み手法
| 手法/`type` | 説明 | 
| --- | --- | 
| `standard` | 目標視認性をアルファマップとしたアルファ合成 | 
| `CrossDissolveContrast`<br>`CrossDissolveColor`<br>`CrossDissolveSaliency` | "Cross Dissolve Without Cross Fade: Preserving Contrast, Color and Salience in Image Compositing"で提案された各手法 | 
| `fukiage2014` | "Visibility-Based Blending for Real-Time Applications"で提案された手法<br>オプションでブラーサイズを指定する| 
| `sandor` | "An augmented reality x-ray system based on visual saliency"で提案された手法 | 
| ネットワーク合成 | `../networks/utils`における`NETWORK_TYPE_LIST`から使用するネットワークを指定する | 
| 視認性予測モデルによる再帰調整 | `../vismodel/vismodel_configs.json`から使用する視認性予測モデルを指定する<br>オプションで使用する損失関数を指定する | 