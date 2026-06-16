import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util
import os

defocus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pre_experiment', 'defocus_matching.py')
spec_defocus = importlib.util.spec_from_file_location('defocus_matching', defocus_path)
defocus_matching = importlib.util.module_from_spec(spec_defocus)
spec_defocus.loader.exec_module(defocus_matching)

stimulus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pre_experiment', 'stimuli_utils.py')
spec_stimulus = importlib.util.spec_from_file_location('stimuli_utils', stimulus_path)
stimuli_utils = importlib.util.module_from_spec(spec_stimulus)
spec_stimulus.loader.exec_module(stimuli_utils)


# ==========================================
# 1. デフォーカスによる背景コントラスト減衰率の計算関数
# (pre-analyze-matching-ar.py から移植)
# ==========================================
_blur_attenuation_cache = {}

def calculate_blur_attenuation_cached(pd_mm, d_fg=50.0, d_bg=150.0, f_center_cpd=4.0):
    """
    瞳孔径(pd_mm)とデフォーカス量に基づいて、背景ノイズのコントラスト減衰率(0.0~1.0)を計算します。
    """
    # --- 使用するパターンとコントラスト計算方法を選択 ---
    # use_michelson = True  # ミケルソンコントラスト（Gabor）を使用する場合
    use_michelson = True # RMSコントラスト（ノイズ）を使用する場合 (デフォルト)

    pd_mm_round = round(pd_mm, 2)
    # 計算方法をキャッシュキーに含める
    cache_key = (pd_mm_round, d_fg, d_bg, use_michelson)
    if cache_key in _blur_attenuation_cache:
        return _blur_attenuation_cache[cache_key]

    if pd_mm_round <= 0:
        return 1.0

    PIXELS_PER_CM = 1 / 0.02331
    ppd_fg = PIXELS_PER_CM * d_fg * math.tan(math.radians(1.0))
    width_deg, height_deg = 7.9, 3.95
    width_fg, height_fg = int(width_deg * ppd_fg), int(height_deg * ppd_fg)

    if use_michelson:
        # Gaborパターンを生成
        base_pattern = stimuli_utils.create_gabor_base(width_fg, height_fg, ppd_fg, f_center_cpd, orientation=0)
    else:
        # ノイズ画像の生成と周波数ドメインでのバンドパスフィルタリング
        np.random.seed(42)
        white_noise = np.random.normal(0, 1, (height_fg, width_fg))
        ft_noise = np.fft.fftshift(np.fft.fft2(white_noise))

        fx = np.fft.fftshift(np.fft.fftfreq(width_fg, d=1/ppd_fg))
        fy = np.fft.fftshift(np.fft.fftfreq(height_fg, d=1/ppd_fg))
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)

        bandwidth_octave = 1.0
        f_min = f_center_cpd / (2 ** (bandwidth_octave / 2))
        f_max = f_center_cpd * (2 ** (bandwidth_octave / 2))
        mask = (R >= f_min) & (R <= f_max)

        ft_filtered = ft_noise * mask
        noise_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(ft_filtered)))

        # 正規化
        max_val = np.max(np.abs(noise_filtered))
        base_pattern = noise_filtered / max_val if max_val > 0 else noise_filtered

    L_bg_temp, C_bg_orig = 15.0, 1.0
    lum_orig = L_bg_temp * (1.0 + C_bg_orig * base_pattern)

    # スケーリングして uint8 画像に変換
    ln_min, ln_max = lum_orig.min(), lum_orig.max()
    if ln_max - ln_min == 0:
        _blur_attenuation_cache[cache_key] = 1.0
        return 1.0

    lum_scaled_uint8 = ((lum_orig - ln_min) / (ln_max - ln_min) * 255.0).astype(np.uint8)
    from PIL import Image
    img_pil = Image.fromarray(lum_scaled_uint8, mode='L')

    # D をディオプトリ差として計算
    d_fg_m = d_fg / 100.0
    d_bg_m = d_bg / 100.0
    D = 0.0 if d_fg_m <= 0 or d_bg_m <= 0 else abs(1.0/d_fg_m - 1.0/d_bg_m)

    blur_img_pil = defocus_matching.apply_torch_fft_blur(img_pil, D, pd_mm_round, ppd_fg)
    blur_scaled_float = np.array(blur_img_pil).astype(np.float32)

    if use_michelson:
        # ミケルソンコントラストで減衰率を計算
        # ブラー適用後の画像を元の輝度スケールに戻す
        lum_blur = blur_scaled_float / 255.0 * (ln_max - ln_min) + ln_min

        def calculate_michelson(arr):
            l_max, l_min = np.max(arr), np.min(arr)
            return (l_max - l_min) / (l_max + l_min) if (l_max + l_min) > 1e-9 else 0.0

        contrast_orig = calculate_michelson(lum_orig)
        contrast_blur = calculate_michelson(lum_blur)
        val = float(contrast_blur / (contrast_orig + 1e-12)) if contrast_orig > 0 else 1.0
    else:
        # RMSコントラストで減衰率を計算
        rms_orig = np.std(lum_scaled_uint8.astype(np.float32))
        rms_blur = np.std(blur_scaled_float)
        val = float(rms_blur / (rms_orig + 1e-12)) if rms_orig > 0 else 1.0

    _blur_attenuation_cache[cache_key] = val
    return val


# ==========================================
# 2. 知覚コントラスト予測モデル群
# ==========================================
class ContrastMatchingModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        # F.softplusを通して正の値に保つため、rawパラメータとして定義
        self.raw_sigma = nn.Parameter(torch.tensor(0.1))
        self.raw_beta = nn.Parameter(torch.tensor(1.0))

    @property
    def sigma(self): return F.softplus(self.raw_sigma) + 1e-6  # 必ず正 (> 0) を保証
    @property
    def beta(self): return F.relu(self.raw_beta)  # 0以上 (>= 0) を保証
    @property
    def gamma(self): return torch.tensor(2.2, device=self.raw_sigma.device)

class ModelA(ContrastMatchingModelBase):
    """ C = S^gamma / (sigma^gamma + beta * M^gamma) """
    def forward(self, S, M, **kwargs):
        num = torch.pow(S + 1e-8, self.gamma)
        den = torch.pow(self.sigma, self.gamma) + self.beta * torch.pow(M + 1e-8, self.gamma)
        return num / den

class ModelB(ContrastMatchingModelBase):
    """ C = S^gamma / (sigma^gamma + beta * f(M)^gamma) """
    def forward(self, S, M, blur_attenuation, **kwargs):
        f_M = M * blur_attenuation
        num = torch.pow(S + 1e-8, self.gamma)
        den = torch.pow(self.sigma, self.gamma) + self.beta * torch.pow(f_M + 1e-8, self.gamma)
        return num / den

class ModelC1(ContrastMatchingModelBase):
    """ C1: B の g(delta_D) を定数 0.2 で代用
        C = S^gamma / (sigma^gamma + beta * (f(M) * 0.2)^gamma) """
    G_CONST = 0.2  # g(delta_D) の代用定数（背景割引）

    def forward(self, S, M, blur_attenuation, **kwargs):
        f_M = M * blur_attenuation
        num = torch.pow(S + 1e-8, self.gamma)
        den = torch.pow(self.sigma, self.gamma) + self.beta * torch.pow(f_M * self.G_CONST + 1e-8, self.gamma)
        return num / den

class ModelC2(ContrastMatchingModelBase):
    """ C2: 左右眼それぞれで B を計算し平均（disparity / 左右blur差の影響を反映）
        C = mean_eyes( S^gamma / (sigma^gamma + beta * f(M)^gamma) ) """
    def forward(self, S, M, blur_attenuation_left, blur_attenuation_right, **kwargs):
        num = torch.pow(S + 1e-8, self.gamma)
        f_M_L = M * blur_attenuation_left
        f_M_R = M * blur_attenuation_right
        den_L = torch.pow(self.sigma, self.gamma) + self.beta * torch.pow(f_M_L + 1e-8, self.gamma)
        den_R = torch.pow(self.sigma, self.gamma) + self.beta * torch.pow(f_M_R + 1e-8, self.gamma)
        return 0.5 * (num / den_L + num / den_R)


# ==========================================
# 3. 学習ループのサンプル
# ==========================================
def train_model(model_class, dataloader, epochs=500, lr=0.01):
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"--- Training {model_class.__name__} ---")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # kwargs形式でモデルに渡すことで、必要な引数のみが使われます
            pred_C = model(**batch)
            loss = criterion(pred_C, batch['C_target'])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f} | "
                  f"sigma: {model.sigma.item():.4f}, beta: {model.beta.item():.4f}, gamma: {model.gamma.item():.4f}")
            
    return model

if __name__ == "__main__":
    # ここにダミーデータセットを作成して train_model() を呼び出す処理を記述できます。
    pass
