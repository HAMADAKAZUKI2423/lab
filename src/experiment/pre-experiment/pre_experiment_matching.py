"""Gabor contrast matching 本実験のエントリーポイント。"""

import tkinter as tk

from matching import MatchingExperimentApp, create_experiment_config


def main() -> None:
    root = tk.Tk()
    MatchingExperimentApp(root, create_experiment_config())
    root.mainloop()


if __name__ == "__main__":
    main()
