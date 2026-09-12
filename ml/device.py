"""Pick the training device without hard-coding CUDA.

Training happens on whatever machine is free — an Apple-silicon laptop via MPS,
a rented GPU droplet via CUDA, or CPU as the last resort. Deployment is
unaffected either way: the server loads the exported ONNX, never torch.
"""

import torch


def pick_device(requested: str = "auto") -> str:
    """Return an Ultralytics device string: "mps", "0" (first CUDA GPU), or "cpu"."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe(device: str) -> str:
    if device == "mps":
        return f"Apple MPS ({torch.backends.mps.is_built() and 'built' or 'unavailable'})"
    if device.isdigit():
        return f"CUDA {torch.cuda.get_device_name(int(device))}"
    return "CPU"
