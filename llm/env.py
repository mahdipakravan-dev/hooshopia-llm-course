"""Environment check: which versions we run and which device training will use.

    python -m llm.env
"""
import platform
import sys


def get_device() -> str:
    """Best available device: CUDA GPU, then Apple Silicon (MPS), then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16(device: str) -> bool:
    """bfloat16 halves memory and speeds up training on modern GPUs."""
    import torch

    return device == "cuda" and torch.cuda.is_bf16_supported()


def main() -> None:
    import numpy
    import torch

    device = get_device()
    print(f"python  {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"torch   {torch.__version__}")
    print(f"numpy   {numpy.__version__}")
    print(f"device  {device}")
    if device == "cuda":
        print(f"gpu     {torch.cuda.get_device_name(0)}")
        print(f"memory  {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"bf16    {supports_bf16(device)}")


if __name__ == "__main__":
    main()
