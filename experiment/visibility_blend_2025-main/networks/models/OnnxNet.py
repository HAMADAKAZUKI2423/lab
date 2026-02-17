import torch
from torch import nn
import onnx
import onnxruntime
import numpy as np
from .INetwork import INetwork

class OnnxNet(INetwork):
    
    def __init__(self):
        super(OnnxNet, self).__init__()
    
    def load_model(self, path):
        
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        # ONNX Runtimeセッションを作成
        self.model = onnxruntime.InferenceSession(path)
    
    def forward(self, x1, x2, tv):
        device = x1.device
        # NumPy配列に変換
        blend_input = x1.cpu().numpy()
        ref_input = x2.cpu().numpy()
        vis_input = tv.cpu().numpy()

        # ONNX Runtimeで推論を実行
        ort_inputs = {
            "ovl": blend_input,
            "bg": ref_input,
            "tv": vis_input
        }
        ort_outs = self.model.run(None, ort_inputs)

        # 結果をPyTorchのテンソルに変換（必要に応じて）
        output_tensor = torch.tensor(ort_outs[0], device=device)

        return output_tensor