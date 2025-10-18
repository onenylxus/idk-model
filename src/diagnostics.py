import platform
import psutil
import torch

def diagnostics() -> None:
    """Print system diagnostics information."""

    print("Running diagnostics...\n")
    print("=" * 64)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {int(round(torch.cuda.get_device_properties(0).total_memory / 1e6)):,d} MB")
        print(f"RAM: {int(round(psutil.virtual_memory().total / 1e6)):,d} MB")
        print(f"Available RAM: {int(round(psutil.virtual_memory().available / 1e6)):,d} MB")
    print(f"CPU cores: {psutil.cpu_count()}")
    print("=" * 64)

if __name__ == "__main__":
    diagnostics()
