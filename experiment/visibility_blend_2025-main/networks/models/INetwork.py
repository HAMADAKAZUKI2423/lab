from torch import nn

TV_INPUT_TYPE_LIST = ["map","scalar"]

class INetwork(nn.Module):
    def __init__(self, tv_input:str = "map"):
        super(INetwork, self).__init__()
        assert tv_input in TV_INPUT_TYPE_LIST
        self.tv_input = tv_input