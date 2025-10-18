# idk-model

Reproduction of large language model (LLM) with `[IDK]` token from **I Don’t Know: Explicit Modeling of Uncertainty with an [IDK] Token** in NeurIPS 2024.

## Requirements

Make sure your computer has the following requirements:

- \>15GB disk space
- \>8GB physical memory (RAM)
- Python v3.10 or later

## Setup

Create a virtual envrionment `venv` and activate it:

```sh
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate # (Mac/Linux)
# or
venv\Scripts\activate # (Windows)
```

> [!IMPORTANT]
> Make sure the following scripts are run inside the virtual environment.

Upgrade pip and install packages:

```sh
# Upgrade pip
python -m pip install --upgrade pip

# 🔥 Install PyTorch
pip install torch torchvision torchaudio # (CPU)
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # (CUDA 12)
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 # (CUDA 13)

# 🤗 Install Hugging Face packages
pip install transformers accelerate bitsandbytes
pip install peft trl
pip install datasets
pip install tokenizers
pip install wandb
pip install hf_xet

# 🎨 Install termcolor
pip install termcolor
```

For **PyTorch**, check CUDA compatibility at [PyTorch's official site](https://pytorch.org/get-started/locally/).

Run the system diagnostics to see if your computer fulfills the requirements, then perform a test inference with **Mistral-7B** (`mistralai/Mistral-7B-v0.1`) base model:

```sh
# Check with system diagnostics
python src/diagnostics.py

# Run a simple inference test
python src/test.py
```

Finally, deactivate the virtual environment when finished to exit:

```sh
deactivate
```

## References

Cohen, R., Dobler, K., Biran, E., & de Melo, G. (2024). **I Don’t Know: Explicit Modeling of Uncertainty with an [IDK] Token.** Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/file/14c018d2e72c521605b0567029ef0efb-Paper-Conference.pdf
