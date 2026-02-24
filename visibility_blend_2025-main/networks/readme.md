# ネットワーク実装
## 使用方法
`./models/INetwork.py`にコントローラーを記載  
`INetwork.tv_input`から目標視認性の入力形式を`map`、`scalar`で識別可能
```
optimized_alphamap = INetwork(target_img, reference_image, target_visibility)
```
`./train/networkTrainer.py`に学習用のクラスを記載

小林知幾修論において、不透明画像で学習されたネットワークは、
`network_configs.json`内の`type:testnet,mode:0,1225_501`である。
半透明画像で学習されたネットワークは、
`network_configs.json`内の`type:testnet,mode:0,trans_501`である。

## 実装ネットワーク
`./utils.py`内の`load_network`関数を参照  
`network_configs.json`に、各ネットワークに対応する重みとそのキーを記載しておく  
`./utils.py`内の`load_network_path`関数で`load`によりロードする重みのキーを指定
| ネットワーク/`type` | 説明 | 
| --- | --- | 
| `alphanet` | 対象画像と参照画像を同一重みのエンコーダーに入力し、目標視認性マップはFiLMエンコーダーに入力するネットワーク | 
| `adaptnet` | 対象画像と参照画像を色チャンネル方向に結合しエンコーダーに入力し、目標視認性マップはFiLMエンコーダーに入力するネットワーク | 
| `mattingnet` | `adaptnet`にASPPを追加したネットワーク| 
| `deeplab` | MobileNetV2をバックボーンとしたDeepLabV3アーキテクチャ | 
| `deeplabv3_res` | ResNet101をバックボーンとしたDeepLabV3アーキテクチャ | 
| `testnet` | パラメータ数を削減したDeepLabV3アーキテクチャ<br>`mode`オプションによりパラメータ数、ASPPありなしを指定 | 
| `scalar_deeplab` | スカラー目標視認性入力の`deeplab` | 
| `scalar_deeplabv3_res` | スカラー目標視認性入力の`scalar_deeplabv3_res` | 
| `scalar_testnet` | スカラー目標視認性入力の`testnet` | 