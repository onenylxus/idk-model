from cprint import print_fail, print_info
from datasets import load_dataset, load_from_disk
from pathlib import Path

# Dataset class
class Dataset:
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

        # If the dataset directory exists, load the dataset
        # otherwise download the dataset and save it
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print_info(f"Found existing dataset '{self.dataset_name}'.")
            self.load_data()
        else:
            self.download_data()
            self.save_data()

    @property
    def data_dir(self) -> Path:
        """Returns the dataset directory path."""

        base_dir = Path(__file__).parent.parent
        data_directory = base_dir / "datasets" / self.dataset_name / self.split
        data_directory.mkdir(parents=True, exist_ok=True)
        return data_directory

    def download_data(self):
        """Downloads the dataset."""

        print_info(f"Downloading dataset '{self.dataset_name}'...")
        self.dataset = load_dataset(self.dataset_name, split=self.split)

    def validate_data(self) -> bool:
        """Validates that the dataset is loaded."""

        if self.dataset is None:
            print_fail("Dataset is not loaded.")
            return False
        return True

    def save_data(self):
        """Saves the dataset to disk."""

        # Ensure dataset is loaded
        if not self.validate_data():
            raise ValueError("Dataset not found.")

        # Save dataset
        print_info(f"Saving dataset to '{self.data_dir}'...")
        self.dataset.save_to_disk(self.data_dir)

    def load_data(self):
        """Returns the loaded dataset."""

        print_info(f"Loading dataset from '{self.data_dir}'...")

        # Load dataset
        self.dataset = load_from_disk(self.data_dir)

# LAMA dataset class
class LamaDataset(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__(dataset_name="facebook/lama", split=split)

# TriviaQA dataset class
class TriviaQaDataset(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__(dataset_name="mandarjoshi/trivia_qa", split=split)

    def download_data(self):
        print_info(f"Downloading dataset '{self.dataset_name}'...")
        self.dataset = load_dataset(self.dataset_name, 'rc', split=self.split)

# PopQA dataset class
class PopQaDataset(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__(dataset_name="akariasai/PopQA", split=split)
