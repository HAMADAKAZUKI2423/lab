"""Image evaluation予備実験のエントリーポイント。"""

import tkinter as tk

from image import ImageExperimentApp, create_image_config


def main() -> None:
    root = tk.Tk()
    ImageExperimentApp(root, create_image_config())
    root.mainloop()


if __name__ == "__main__":
    main()
