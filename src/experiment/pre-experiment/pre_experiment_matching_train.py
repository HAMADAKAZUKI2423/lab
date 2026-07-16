"""Gabor contrast matching trainingのエントリーポイント。"""

import tkinter as tk

from matching import MatchingExperimentApp, create_training_config


def main() -> None:
    root = tk.Tk()
    MatchingExperimentApp(root, create_training_config())
    root.mainloop()


if __name__ == "__main__":
    main()
