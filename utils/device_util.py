import torch


class Device:
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.value = torch.device("cuda" if use_cuda else "cpu")


cur_device = Device().value