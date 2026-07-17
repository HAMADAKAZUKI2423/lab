"""Tkinter実験アプリで共有する最小限のUI状態と補助処理。"""

import tkinter as tk


class ExperimentBaseUI:
    """実験固有の画面遷移を持たない、最小限のUI基盤。"""

    def __init__(self, root: tk.Tk):
        self.root = root

        # 参加者情報と位置合わせで共通して使うTkinter変数
        self.participant_id = tk.StringVar()
        self.participant_age = tk.StringVar()
        self.participant_gender = tk.StringVar()
        self.participant_ipd = tk.StringVar()
        self.participant_dominance = tk.StringVar(value="Right")
        self.offset_x = tk.IntVar(value=0)
        self.offset_y = tk.IntVar(value=0)
        self.evaluation_val = tk.IntVar(value=3)

        # 共通のUI状態
        self.key_bindings: dict[str, str] = {}
        self.participant_frame = None
        self.ctrl_frame = None
        self.instruction_frame = None
        self.eval_frame = None

    def clear_key_bindings(self) -> None:
        """このアプリがrootへ登録したキーバインドを解除する。"""
        for key, binding_id in list(self.key_bindings.items()):
            try:
                self.root.unbind(key, binding_id)
            except Exception:
                pass
        self.key_bindings.clear()

    def _destroy_frame(self, attribute: str) -> None:
        """指定属性のFrameが存在する場合だけ安全に破棄する。"""
        frame = getattr(self, attribute, None)
        if frame is not None and frame.winfo_exists():
            frame.destroy()
        setattr(self, attribute, None)

    def adjust_offset(self, dx: int, dy: int):
        """位置合わせオフセットを更新し、表示へ反映する。"""
        self.offset_x.set(self.offset_x.get() + dx)
        self.offset_y.set(self.offset_y.get() + dy)
        self.update_calibration_view()
        return "break"

    def update_calibration_view(self, *args) -> None:
        """各実験固有のキャリブレーション表示を更新する。"""
        raise NotImplementedError
