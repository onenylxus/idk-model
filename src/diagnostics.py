from cprint import print_info, print_warn
from imodel import MistralBaseModel
import platform
import psutil
import torch

def diagnostics():
    """Print system diagnostics information."""

    python_version = platform.python_version()
    is_cuda = torch.cuda.is_available()
    gpu_memory = int(round(torch.cuda.get_device_properties(0).total_memory / 1e6)) if is_cuda else 0
    ram_total = int(round(psutil.virtual_memory().total / 1e6))
    ram_avail = int(round(psutil.virtual_memory().available / 1e6))

    label_w = 16
    print_info("Running diagnostics...")
    print("\n" + "=" * 64)
    print(f"{'OS:':<{label_w}} {platform.system()} {platform.release()}")
    print(f"{'Python:':<{label_w}} {python_version}")
    print(f"{'PyTorch:':<{label_w}} {torch.__version__}")
    print(f"{'CUDA available:':<{label_w}} {is_cuda}")
    if is_cuda:
        print(f"{'⤷ CUDA version:':<{label_w}} {torch.version.cuda}")
        print(f"{'⤷ GPU:':<{label_w}} {torch.cuda.get_device_name(0)}")
        print(f"{'⤷ GPU memory:':<{label_w}} {gpu_memory:,d} MB")
    print(f"{'RAM:':<{label_w}} {ram_total:,d} MB")
    print(f"{'Available RAM:':<{label_w}} {ram_avail:,d} MB")
    print(f"{'CPU:':<{label_w}} {platform.processor()}")
    print(f"{'CPU cores:':<{label_w}} {psutil.cpu_count()}")
    print("=" * 64 + "\n")

    if python_version < "3.10":
        print_warn("Python version is below 3.10, which may lead to compatibility issues.")

    if is_cuda and gpu_memory < 8192:
        print_warn("GPU memory is below 8GB, which may lead to performance issues.")
    elif not is_cuda and ram_avail < 8192:
        print_warn("Available RAM is below 8GB, which may lead to performance issues.")

if __name__ == "__main__":
    diagnostics()
    model = MistralBaseModel()
    model.generate("Hello, my name is")
