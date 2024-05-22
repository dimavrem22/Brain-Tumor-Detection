import torch

def get_device():
    """
    Get the device to use for training (cuda if available, then mps, then cpu)
    """
    if torch.cuda.is_available():
        print("Running on CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running on MPS")
        return torch.device("mps")
    else:
        print("Running on CPU")
        return torch.device("cpu")
