from cprint import print_answer, print_fail, print_info, print_pass, print_warn
from idataset import LamaGoogleReDataset, LamaTrexDataset, LamaSquadDataset, PopQaDataset, TriviaQaDataset
from imodel import IModel, BertBaseModel, MistralBaseModel, PythiaSmallBaseModel, PythiaLargeBaseModel
from tqdm import tqdm
from utils import is_correct_prediction
import random

# Tester class
class ITester:
    def __init__(self, model, dataset):
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

        tp_count = 0 # Correct predictions
        fp_count = 0 # Incorrect non-[IDK] predictions
        total_count = 0

        for sample in tqdm(dataset, desc=f"{self.model.name} -> {self.dataset.name}"):
            print()
            try:
                is_correct, is_confident = self.evaluate_impl(sample)
                total_count += 1
                print()

                if is_correct:
                    print_pass(f"Sample #{total_count} is correct.")
                    tp_count += 1
                else:
                    print_warn(f"Sample #{total_count} is incorrect.")
                    if is_confident:
                        fp_count += 1
            except Exception as e:
                print_fail(f"Error evaluating sample #{total_count}: {e}")
                is_correct = False
            print()

        accuracy = (tp_count / total_count) * 100 if total_count > 0 else 0.0
        precision = (tp_count / (tp_count + fp_count)) * 100 if (tp_count + fp_count) > 0 else 0.0
        recall = (tp_count / total_count) * 100 if total_count > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print_info("Evaluation completed.")

        label_w = 16
        print("\n" + "=" * 64)
        print(f"{'Model:':<{label_w}} {self.model.name}")
        print(f"{'Dataset:':<{label_w}} {self.dataset.name}")
        print(f"{'Count:':<{label_w}} {tp_count}/{total_count}")
        print(f"{'Accuracy:':<{label_w}} {accuracy:.2f}%")
        print(f"{'Precision (P):':<{label_w}} {precision:.2f}%")
        print(f"{'Recall (R):':<{label_w}} {recall:.2f}%")
        print(f"{'F1 Score (F1):':<{label_w}} {f1_score:.2f}%")
        print("=" * 64 + "\n")

        return accuracy, tp_count, total_count

    def evaluate_impl(self, sample):
        """Implementation of the evaluation logic. The first boolean indicates if the prediction is correct, the second boolean indicates if the model was confident."""

        prompt = sample["prompt"]
        gold = sample["answer"]

        # Generate prediction
        try:
            prediction = self.model.generate(prompt)
            print_answer(gold)
            return is_correct_prediction(prediction, gold), prediction.lower()[0] != "[idk]"
        except Exception as e:
            print_fail(f"Error generating prediction for prompt '{prompt}': {e}")
            return False

# LAMA (Google-RE) dataset tester class
class LamaGoogleReTester(ITester):
    def __init__(self, model: IModel):
        dataset = LamaGoogleReDataset("test")
        super().__init__(model, dataset)

# LAMA (TREx) dataset tester class
class LamaTrexTester(ITester):
    def __init__(self, model: IModel):
        dataset = LamaTrexDataset("test")
        super().__init__(model, dataset)

# LAMA (Squad) dataset tester class
class LamaSquadTester(ITester):
    def __init__(self, model: IModel):
        dataset = LamaSquadDataset("test")
        super().__init__(model, dataset)

# TriviaQA dataset tester class
class TriviaQaTester(ITester):
    def __init__(self, model: IModel):
        dataset = TriviaQaDataset("validation")
        super().__init__(model, dataset)

# PopQA dataset tester class
class PopQaTester(ITester):
    def __init__(self, model: IModel):
        dataset = PopQaDataset("validation")
        super().__init__(model, dataset)

if __name__ == "__main__":
    models = [
        # BertBaseModel(),
        MistralBaseModel(),
        # PythiaSmallBaseModel(),
        # PythiaLargeBaseModel()
    ]

    testers = []
    for model in models:
        testers.extend([
            LamaGoogleReTester(model),
            LamaTrexTester(model),
            LamaSquadTester(model),
            TriviaQaTester(model),
            PopQaTester(model),
        ])

    for tester in testers:
        tester.evaluate(max_samples=10000)
