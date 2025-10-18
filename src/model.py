from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any, Final, Literal
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
class Hyperparameters:
    max_new_tokens: Final[int] = 50
    temperature: Final[float] = 0.7
    top_k: Final[int] = 50
    top_p: Final[float] = 0.95

# Base model class
class Model:
    def __init__(
        self,
        name: str,
        pretrained_model: str,
        dtype: Any = torch.bfloat16
    ):
        self.name = name
        self.pretrained_model = pretrained_model
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

        # If the checkpoint directory exists, load the model
        # otherwise download the model and save it
        if self.chpt_dir.exists() and any(self.chpt_dir.iterdir()):
            self.load_model()
        else:
            self.download_model()
            self.save_model()

        print(f"Model '{self.name}' with device type {device} is ready.")

    @property
    def chpt_dir(self) -> Path:
        """Returns the checkpoint directory path."""

        base_dir = Path(__file__).parent / ".."
        chpt_directory = base_dir / "checkpoints" / self.name
        chpt_directory.mkdir(parents=True, exist_ok=True)
        return chpt_directory

    # Device map
    @property
    def device_map(self) -> Literal["auto"] | None:
        """Returns the device map based on availability of CUDA."""

        return "auto" if device.type == "cuda" else None

    def download_model(self):
        """Downloads the tokenizer and model."""

        print(f"Downloading model '{self.pretrained_model}'...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device_map
        )

    def save_model(self):
        """Saves the tokenizer and model to the checkpoint directory."""

        print(f"Saving model to '{self.chpt_dir}'...")
        self.tokenizer.save_pretrained(self.chpt_dir)
        self.model.save_pretrained(self.chpt_dir)

    def load_model(self):
        """Loads the tokenizer and model from the checkpoint directory."""

        print(f"Loading model from '{self.chpt_dir}'...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.chpt_dir,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.chpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device_map
        )

    def generate(self, prompt: str) -> str:
        """Generates text based on the given prompt."""

        # Ensure model and tokenizer are loaded
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        # Tokenize input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=Hyperparameters.max_new_tokens,
            temperature=Hyperparameters.temperature,
            top_k=Hyperparameters.top_k,
            top_p=Hyperparameters.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

# Mistral-7B base model class
class MistralBaseModel(Model):
    def __init__(self):
        super().__init__(
            name="mistral-7b-base",
            pretrained_model="mistralai/Mistral-7B-v0.1"
        )
