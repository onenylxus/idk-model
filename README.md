# idk-model

Reproduction of large language model (LLM) with `[IDK]` token from **I Don’t Know: Explicit Modeling of Uncertainty with an [IDK] Token** in NeurIPS 2024.

## Requirements

Make sure your computer has the following requirements:

- \>40GB disk space
  - ~15GB for Mistral-7B model
  - ~12GB for Wikipedia dataset, and ~5GB for testing datasets
  - ~8GB for checkpoints of base and trained models
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
pip install tqdm
pip install hf_xet

# 🎨 Install termcolor
pip install termcolor
```

For **PyTorch**, check CUDA compatibility at [PyTorch's official site](https://pytorch.org/get-started/locally/).

**LAMA** datasets cannot be downloaded directly using HuggingFace, so we manually download the zipped source folder into a temporary directory:

```sh
mkdir temp
pushd temp
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
mv data lama
rm data.zip
popd
```

Run the system diagnostics to see if your computer fulfills the requirements, then perform a test inference with **Mistral-7B** (`mistralai/Mistral-7B-v0.1`) base model:

```sh
python src/diagnostics.py
```

Basic training and testing can be done by running other Python files:

```sh
# Train
python src/itrainer.py

# Test
python src/itester.py
```

Finally, deactivate the virtual environment when finished to exit:

```sh
deactivate
```

## Remarks

For detailed demonstration, using the repository referenced in the research paper produces more accurate results. However, some libraries and dataset sources are outdated and the reference repository is not actively maintained.

## References

Cohen, R., Dobler, K., Biran, E., & de Melo, G. (2024). **I Don’t Know: Explicit Modeling of Uncertainty with an [IDK] Token.** Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/file/14c018d2e72c521605b0567029ef0efb-Paper-Conference.pdf

Cohen, R.. **roi-hpi/IDK-token-tuning.** GitHub. https://github.com/roi-hpi/IDK-token-tuning
