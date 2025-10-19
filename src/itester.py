from cprint import print_fail, print_info, print_pass, print_warn
from idataset import IDataset, LamaTrexDataset
from imodel import IModel, BertBaseModel, MistralBaseModel, PythiaSmallBaseModel, PythiaLargeBaseModel
from tqdm import tqdm
import random
import re

# Tester class
class ITester:
    def __init__(self, model: IModel, dataset: IDataset):
        self.model = model
        self.dataset = dataset

    def evaluate(self, max_samples: int=None) -> tuple[float, int, int]:
        """Evaluates the model on the dataset."""

        # Ensure model and dataset are loaded
        if not self.model.validate_model():
            raise ValueError("Model is not loaded.")
        if not self.dataset.validate_data():
            raise ValueError("Dataset is not loaded.")

        dataset = self.dataset.dataset
        if max_samples is not None:
            n = min(max_samples, len(dataset))
            indices = random.sample(range(len(dataset)), n)
            dataset = dataset.select(indices)

        print_info(f"Starting evaluation: model {self.model.name} on dataset {self.dataset.name}...")

        correct_count = 0
        total_count = 0

        for sample in tqdm(dataset, desc="{self.model.name} -> {self.dataset.name}"):
            total_count += 1
            try:
                is_correct = self.evaluate_impl(sample)
                if is_correct:
                    print_pass(f"Sample #{total_count} is correct.")
                else:
                    print_warn(f"Sample #{total_count} is incorrect.")
            except Exception as e:
                print_fail(f"Error evaluating sample #{total_count}: {e}")
                is_correct = False

            if is_correct:
                correct_count += 1

        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        print_pass("Evaluation completed.")

        label_w = 16
        print("\n" + "=" * 64)
        print(f"{'Model:':<{label_w}} {self.model.name}")
        print(f"{'Dataset:':<{label_w}} {self.dataset.name}")
        print(f"{'Count:':<{label_w}} {correct_count}/{total_count}")
        print(f"{'Accuracy:':<{label_w}} {accuracy:.2f}%")
        print("=" * 64 + "\n")

        return accuracy, correct_count, total_count

    def evaluate_impl(self, sample) -> bool:
        """Implementation of the evaluation logic."""

        # Implementation depends of the specific dataset
        pass

# LAMA (TREx) dataset tester class
class LamaTrexTester(ITester):
    def __init__(self, model: IModel):
        dataset = LamaTrexDataset("test")
        super().__init__(model, dataset)

    def evaluate_impl(self, sample) -> bool:
        # Prepare prompt
        prompt = sample["masked_sentence"].strip() + " Replace [MASK] with only one word:"

        # Generate prediction
        try:
            first_word = lambda s: re.search(r"[A-Za-z]+", s).group(0) if re.search(r"[A-Za-z]+", s) else s.strip().split()[0] if s.strip().split() else ""

            prediction = self.model.generate(prompt)
            model_answer = first_word(prediction)
            correct_answer = first_word(sample["obj_surface"])

            label_w = 16
            print("\n" + "=" * 64)
            print(f"{'Model Answer:':<{label_w}} {model_answer}")
            print(f"{'Correct Answer:':<{label_w}} {correct_answer}")
            print("=" * 64 + "\n")

            return model_answer.lower() == correct_answer.lower()
        except Exception as e:
            print_fail(f"Error generating prediction for prompt '{prompt}': {e}")
            return False

if __name__ == "__main__":
    models = [
        # BertBaseModel(),
        MistralBaseModel(),
        # PythiaSmallBaseModel(),
        # PythiaLargeBaseModel()
    ]

    testers = [
        LamaTrexTester(model) for model in models
    ]

    for tester in testers:
        tester.evaluate(max_samples=10000)
