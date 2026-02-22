from PIL import Image
import random
import os

def generate_solid_color_image(width, height, filename="solid_color.png"):
    """
    指定されたサイズで、ランダムな単一色の画像を生成し保存します。
    また、その色のRGB値をコンソールに出力します。
    """
    
    # 1. ランダムなRGB値を生成 (0〜255の範囲)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color_rgb = (r, g, b)

    # 2. 画像を生成する
    # mode='RGB': カラー画像
    # size=(width, height): 画像の大きさ
    # color=color_rgb: 塗りつぶす色
    img = Image.new(mode='RGB', size=(width, height), color=color_rgb)

    # 3. RGB値をコンソールに出力
    print("-" * 30)
    print("【生成結果】")
    print(f"画像サイズ: {width} x {height}")
    print(f"RGB値      : R={r}, G={g}, B={b}")
    print("-" * 30)

    # 4. 画像をファイルに保存
    try:
        img.save(filename)
        # 保存先の絶対パスを取得して表示（わかりやすくするため）
        full_path = os.path.abspath(filename)
        print(f"画像を保存しました: {full_path}")
    except Exception as e:
        print(f"画像の保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    # 画像のサイズ設定
    IMG_WIDTH = 400
    IMG_HEIGHT = 300
    # 保存するファイル名
    FILE_NAME = "output_400x300.png"

    # 関数を呼び出して画像を生成
    generate_solid_color_image(IMG_WIDTH, IMG_HEIGHT, FILE_NAME)