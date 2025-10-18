from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

class Model:
    def __init__(self, name: str, pretrained_model: str):
        self.name = name
        self.pretrained_model = pretrained_model
        self.tokenizer = None

    @property
    def chpt_dir(self) -> Path:
        base_dir = Path(__file__).parent / ".."
        chpt_directory = base_dir / "checkpoints" / self.name
        chpt_directory.mkdir(parents=True, exist_ok=True)
        return chpt_directory

    def download(self):
        print(f"Downloading model '{self.pretrained_model}'...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if device.type == "cuda" else None
        )

        print(f"Saving model to '{self.chpt_dir}'...")
        self.model.save_pretrained(self.chpt_dir)
        self.tokenizer.save_pretrained(self.chpt_dir)

    def load(self):
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
            device_map="auto" if device.type == "cuda" else None
        )

        print("Model loaded successfully.")

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

if __name__ == "__main__":
    model = Model(name="mistral-7b-base", pretrained_model="mistralai/Mistral-7B-v0.1")
    if not model.chpt_dir.exists() or not any(model.chpt_dir.iterdir()):
        model.download()
    model.load()
    prompt = "Hello, my name is"
    response = model.generate(prompt)
    print(f"Prompt: {prompt}\n\nResponse: {response}")
