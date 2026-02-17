import numpy as np

# 1. データの定義（行ごとに「ある色を撮影した時のRGB値」を記述）
# Input (Before Glass): 
# 行 = [R成分, G成分, B成分]
S_rows = np.array([
    [56.11, 8.35, 4],       # Rを表示した時の測定値
    [14.3, 53.92, 23.03],   # Gを表示した時の測定値
    [0.98, 8.36, 42.85]     # Bを表示した時の測定値
])



# Output (After Glass):
# 対応する「ガラスを通した時」の測定値
D_rows = np.array([
    [10.17, 1.15, 0.26],    # Rを表示(ガラスなし)
    [2.31, 11.93, 4.84],    # Gを表示(ガラスなし)
    [0.04, 1.74, 11.36]    # Bを表示(ガラスあり)
])



# 2. 計算のための準備
# 通常、数式 v_out = M * v_in ではベクトルを「縦」に扱うため、転置(.T)します。
# これで各「列」が1つの色サンプルになります。
S = S_rows.T
D = D_rows.T

# 3. 行列の算出: M = D * S^-1
try:
    S_inv = np.linalg.inv(S)
    M = D @ S_inv  # 行列積
    
    print("--- 算出した変換行列 M (3x3) ---")
    np.set_printoptions(precision=4, suppress=True)
    print(M)
    print("\n")

    # 4. 検算
    print("--- 検算: Input(R表示時) -> Output(R表示時) ---")
    # 元の「Rを表示した時の測定値」を入力してみる
    test_input = np.array([56.11, 8.35, 4])  # R表示時の入力値
    
    # 行列Mに入力を掛ける (M @ v)
    predicted_output = M @ test_input
    
    print(f"Input (Before): {test_input}")
    print(f"Actual (After): {D_rows[0]}") # 期待値 [66, 11, 6]
    print(f"Predicted:      {np.round(predicted_output).astype(int)}")

except np.linalg.LinAlgError:
    print("エラー: 逆行列が計算できません。")