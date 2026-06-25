import numpy as np

# =============================================================
# 入力: 各行が R, G, B / 各列が Y(輝度), x, y の 3x3 行列
# 計測値をここに入れてください
# =============================================================
BG  = np.array([   # 1. BG  (生背景)
    [39.73, 0.6427, 0.3282],   # R: (Y, x, y)
    [150.57, 0.3074, 0.5959],   # G: (Y, x, y)
    [12.73,  0.1546, 0.052],   # B: (Y, x, y)
])
BGT = np.array([   # 2. BGT (背景の透過)
    [7.17, 0.6289, 0.3252],
    [31.60, 0.2942, 0.5892],
    [3.10, 0.1547, 0.0467],
])
FG  = np.array([   # 3. FG  (生前景)
    [30.50, 0.6570, 0.3206],
    [114.00, 0.3290, 0.6140],
    [11.17, 0.1595, 0.0496],
])
FGR = np.array([   # 4. FGR (前景の反射)
    [20.13, 0.6450, 0.3324],
    [73.50, 0.3194, 0.6238],
    [6.80, 0.1524, 0.0515],
])


def Yxy_to_XYZ(row):
    """1行 (Y, x, y) を (X, Y, Z) に変換"""
    Y, x, y = row
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z])


def to_XYZ_columns(M_Yxy):
    """
    行=R,G,B / 列=Y,x,y の行列を受け取り、
    各原色を XYZ に変換したうえで「列=原色(R,G,B), 行=X,Y,Z」の行列にして返す。
    （v' = M v の列ベクトル規約で扱うため転置して列に並べる）
    """
    rows_xyz = np.array([Yxy_to_XYZ(M_Yxy[i]) for i in range(3)])  # 行=R,G,B, 列=X,Y,Z
    return rows_xyz.T  # 列=R,G,B, 行=X,Y,Z


# --- XYZ へ変換（原色を列に並べた行列）---
BG_xyz  = to_XYZ_columns(BG)
BGT_xyz = to_XYZ_columns(BGT)
FG_xyz  = to_XYZ_columns(FG)
FGR_xyz = to_XYZ_columns(FGR)

# --- 変換行列 T, R を計算 ---
#   BGT_xyz = T @ BG_xyz   ->  T = BGT_xyz @ inv(BG_xyz)
#   FGR_xyz = R @ FG_xyz   ->  R = FGR_xyz @ inv(FG_xyz)
T = BGT_xyz @ np.linalg.inv(BG_xyz)
R = FGR_xyz @ np.linalg.inv(FG_xyz)

# --- T = R C を満たす C を C = R^{-1} T で導出 ---
C = np.linalg.inv(R) @ T

# =============================================================
# 結果表示
# =============================================================
np.set_printoptions(precision=6, suppress=True)
print("BG_xyz (列=R,G,B / 行=X,Y,Z):\n", BG_xyz)
print("BGT_xyz:\n", BGT_xyz)
print("FG_xyz:\n", FG_xyz)
print("FGR_xyz:\n", FGR_xyz)
print("\nT (BG -> BGT, 透過の変換行列):\n", T)
print("\nR (FG -> FGR, 反射の変換行列):\n", R)
print("\nC = R^{-1} T:\n", C)

# --- 検算 ---
print("\n[検算] R @ C と T の差(最大):", np.max(np.abs(R @ C - T)))