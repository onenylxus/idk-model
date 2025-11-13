from cprint import print_fail, print_info, print_pass, print_prompt, print_response
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base model class
class IModel:
    def __init__(
        self,
        name,
        pretrained_model,
        dtype = torch.bfloat16
    ):
        self.name = name
        self.pretrained_model = pretrained_model
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

        # If the checkpoint directory exists, load the model
        # otherwise download the model and save it
        if self.chpt_dir.exists() and any(self.chpt_dir.iterdir()):
            print_info(f"Found existing checkpoint for model '{self.name}'.")
            self.load_model()
        else:
            self.download_model()
            self.save_model()

        print_pass(f"Model '{self.name}' with device type {device} is ready.")

    @property
    def chpt_dir(self):
        """Returns the checkpoint directory path."""

        base_dir = Path(__file__).parent.parent
        chpt_directory = base_dir / "checkpoints" / self.name
        chpt_directory.mkdir(parents=True, exist_ok=True)
        return chpt_directory

    # Device map
    @property
    def device_map(self):
        """Returns the device map based on availability of CUDA."""

        return "auto" if device.type == "cuda" else None

    def download_model(self):
        """Downloads the tokenizer and model."""

        print_info(f"Downloading model '{self.pretrained_model}'...")

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

    def validate_model(self):
        """Validates that the model and tokenizer are loaded."""

        if self.tokenizer is None:
            print_fail("Tokenizer is not loaded.")
            return False
        if self.model is None:
            print_fail("Model is not loaded.")
            return False
        return True

    def save_model(self):
        """Saves the tokenizer and model to the checkpoint directory."""

        # Ensure model and tokenizer are loaded
        if not self.validate_model():
            raise ValueError("Model and tokenizer not found.")

        # Save tokenizer and model
        print_info(f"Saving model to '{self.chpt_dir}'...")
        self.tokenizer.save_pretrained(self.chpt_dir)
        self.model.save_pretrained(self.chpt_dir)

    def load_model(self):
        """Loads the tokenizer and model from the checkpoint directory."""

        print_info(f"Loading model from '{self.chpt_dir}'...")

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

    def generate(self, prompt, display = True):
        """Generates text based on the given prompt."""

        # Ensure model and tokenizer are loaded
        if not self.validate_model():
            raise ValueError("Model and tokenizer must be loaded before generation.")

        if display:
            print_prompt(prompt)

        # Tokenize input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = prompt_response[len(prompt):].strip()

        if display:
            print_response(response)
        return response

    def predict(self, prompt, display = True):
        """Generates a prediction based on the given prompt."""

        # Ensure model and tokenizer are loaded
        if not self.validate_model():
            raise ValueError("Model and tokenizer must be loaded before generation.")

        # Validate tokenizer has mask token
        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer does not have a [MASK] token. Use generate() instead.")

        if display:
            print_prompt(prompt)

        # Tokenize input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # Generate output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Extract the prediction for the [MASK] token
        mask_token_logits = logits[0, mask_token_index, :]
        top_token_id = mask_token_logits.argmax(dim=-1, keepdim=True)
        prediction = self.tokenizer.decode(top_token_id)[0]

        if display:
            print_response(prediction)
        return prediction

# BERT base model (cased) class
class BertBaseModel(IModel):
    def __init__(self):
        super().__init__(
            name="bert-base-cased-base",
            pretrained_model="google-bert/bert-base-cased"
        )

# Mistral-7B base model class
class MistralBaseModel(IModel):
    def __init__(self):
        super().__init__(
            name="mistral-7b-base",
            pretrained_model="mistralai/Mistral-7B-v0.1"
        )

# Pythia-70M base model class
class PythiaSmallBaseModel(IModel):
    def __init__(self):
        super().__init__(
            name="pythia-70m-base",
            pretrained_model="EleutherAI/pythia-70m"
        )

# Pythia-2.8B base model class
class PythiaLargeBaseModel(IModel):
    def __init__(self):
        super().__init__(
            name="pythia-2.8b-base",
            pretrained_model="EleutherAI/pythia-2.8b"
        )
