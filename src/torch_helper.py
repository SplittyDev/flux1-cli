import torch


def get_best_device(force_cpu: bool = False) -> torch.device:
    """
    Attempt to find the fastest device for inference.
    """

    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend (GPU).")
    elif not force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend (GPU).")
    else:
        device = torch.device("cpu")
        if force_cpu:
            print("Using CPU backend.")
        else:
            print("No GPU detected. Using CPU as fallback.")
    return device
