from cprint import print_fail, print_info, print_pass
from datasets import Dataset, load_dataset, load_from_disk
from pathlib import Path
import glob
import json

# Dataset class
class IDataset:
    def __init__(self, name: str, dataset_name: str, split: str = "train"):
        self.name = name
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

        # If the dataset directory exists, load the dataset
        # otherwise download the dataset and save it
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print_info(f"Found existing dataset '{self.name}'.")
            self.load_data()
        else:
            self.download_data()
            self.save_data()

        print_pass(f"Dataset '{self.name}' is ready.")

    @property
    def data_dir(self) -> Path:
        """Returns the dataset directory path."""

        base_dir = Path(__file__).parent.parent
        data_directory = base_dir / "datasets" / self.name / self.split
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

# LAMA dataset base class
class LamaDataset(IDataset):
    def __init__(self, config: str, temp_folder: str, split: str = "train"):
        self.config = config
        self.temp_folder = temp_folder
        super().__init__(name=f"lama_{config}", dataset_name="facebook/lama", split=split)

        # Check if temporary directory exists
        if not self.temp_dir.exists() or not any(self.temp_dir.iterdir()):
            raise ValueError(f"Temporary directory '{self.temp_dir}' does not exist or is empty. ")

    @property
    def temp_dir(self) -> Path:
        """Returns the temporary directory path where the dataset holds."""

        base_dir = Path(__file__).parent.parent
        temp_directory = base_dir / "temp" / 'lama' / self.temp_folder
        return temp_directory

    def download_data(self):
        print_info(f"Downloading dataset '{self.dataset_name}' with config {self.config}...")

        # Collect relations from JSONL file
        relations = []
        with open(self.temp_dir / ".." / "relations.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                relations.append(obj)

        jsonl_files = glob.glob(str(self.temp_dir / "*.jsonl"))
        raw_data = []
        data = []

        # Collect raw data from JSONL files
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line.strip()))

        # Process raw data into final format
        for obj in raw_data:
            # Build base entry
            entry = {}
            for k, v in obj.items():
                if k == "evidences":
                    continue
                entry[k] = v

            # Join data with relation
            for relation in relations:
                if relation["relation"] == obj["predicate_id"]:
                    for k, v in relation.items():
                        if k == "relation":
                            continue
                        entry[k] = v
                    break

            for evidence in obj["evidences"]:
                sample = entry.copy()
                sample.update(evidence)
                data.append(sample)

        # Validate data
        if len(data) == 0:
            raise ValueError(f"No data found in temporary directory '{self.temp_dir}'.")

        # Convert to Hugging Face dataset
        self.dataset = Dataset.from_list(data)


# LAMA (Google-RE) dataset class
class LamaGoogleReDataset(LamaDataset):
    def __init__(self, split: str = "train"):
        super().__init__(config="google_re", temp_folder="Google_RE", split=split)

# LAMA (TREx) dataset class
class LamaTrexDataset(LamaDataset):
    def __init__(self, split: str = "train"):
        super().__init__(config="trex", temp_folder="TREx", split=split)

# LAMA (Squad) dataset class
class LamaSquadDataset(LamaDataset):
    def __init__(self, split: str = "train"):
        super().__init__(config="squad", temp_folder="Squad", split=split)

# TriviaQA dataset class
class TriviaQaDataset(IDataset):
    def __init__(self, split: str = "train"):
        super().__init__(name="trivia_qa", dataset_name="mandarjoshi/trivia_qa", split=split)

    def download_data(self):
        print_info(f"Downloading dataset '{self.dataset_name}'...")
        self.dataset = load_dataset(self.dataset_name, 'rc', split=self.split)

# PopQA dataset class
class PopQaDataset(IDataset):
    def __init__(self, split: str = "train"):
        super().__init__(name="pop_qa", dataset_name="akariasai/PopQA", split=split)
