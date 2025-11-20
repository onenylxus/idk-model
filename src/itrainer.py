from cprint import print_info
from torch import nn
from transformers import Trainer
import numpy as np
import torch
import wandb

wandb.init(project="idk-model")

# Trainer class
class ITrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_training_idk = kwargs.get('is_training_idk', True)
        self.idk_weight_max = kwargs.get('idk_weight_max', 0.5)
        self.idk_weight_schedule = kwargs.get('idk_weight_schedule', 'constant') # 'constant' | 'increasing' | 'decreasing' | 'adaptive'
        self.num_expected_steps = kwargs.get('num_expected_steps', 100000)
        self.correct_prediction_regularization = kwargs.get('correct_prediction_regularization', False)

        self.idk_token_index = self.data_collator.tokenizer.convert_tokens_to_ids('[IDK]')
        self.idk_weight_current = 0.0

    def get_idk_weight_scheduler(self):
        """Get the current IDK weight based on the training step and schedule."""

        global_step = self.state.global_step

        if self.idk_weight_schedule == 'constant':
            return self.idk_weight_max / 2
        elif self.idk_weight_schedule == 'increasing':
            return self.idk_weight_max * np.tanh((global_step + self.num_expected_steps / 20) / (self.num_expected_steps / 2))
        elif self.idk_weight_schedule == 'decreasing':
            return self.idk_weight_max * (1 - np.tanh(global_step / (self.num_expected_steps / 2)))

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss, with special handling for IDK training."""

        if not self.is_training_idk:
            wandb.log({"idk_weight": 0.0}, step=self.state.global_step)
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        else:
            return self.compute_loss_idk(model, inputs, return_outputs)

    def compute_loss_idk(self, model, inputs, return_outputs=False):
        """Compute the loss with IDK training."""

        # Update IDK weight every 1000 steps
        if self.state.global_step % 1000 == 0:
            self.idk_weight_current = self.get_idk_weight_scheduler()
            print_info(f"IDK weight updated to: {self.idk_weight_current:.4f} with global step {self.state.global_step} and schedule {self.idk_weight_schedule}")

        # Use model to predict logits
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        vocab_size = logits.shape[2]

        masked_token = labels != -100
        masked_logits = logits[masked_token]
        predicted_logits = masked_logits.argmax(dim=-1)
        masked_token_gt = labels[masked_token]

        correct_predictions = predicted_logits == masked_token_gt
        masked_correct_logits = masked_logits[correct_predictions]
        masked_incorrect_logits = masked_logits[~correct_predictions]
        one_hot_labels = nn.functional.one_hot(masked_token_gt, num_classes=vocab_size)

        # Apply regular cross entropy for correct predictions
        one_hot_correct_predictions = one_hot_labels[correct_predictions]
        if self.correct_prediction_regularization:
            loss_correct = self.cross_entropy_loss_regularized(masked_correct_logits, one_hot_correct_predictions)
        else:
            loss_correct = self.cross_entropy_loss(masked_correct_logits, one_hot_correct_predictions)

        # Apply IDK loss for incorrect predictions
        one_hot_incorrect_predictions = one_hot_labels[~correct_predictions]

        if self.idk_weight_schedule == 'adaptive':
            prob_wrong_predictions = nn.functional.softmax(masked_incorrect_logits, dim=-1)

            gold_token_index = masked_token_gt[~correct_predictions]
            prob_gold_tokens = prob_wrong_predictions[torch.arange(prob_wrong_predictions.shape[0]), gold_token_index]
            prob_top_token = prob_wrong_predictions.max(dim=-1)[0]
            idk_weights = self.idk_weight_max * (torch.ones_like(prob_top_token) - prob_gold_tokens / prob_top_token)

            one_hot_incorrect_labels = one_hot_incorrect_predictions * (1 - idk_weights.unsqueeze(-1))
            one_hot_incorrect_labels[:, self.idk_token_index] = idk_weights
            loss_incorrect = self.cross_entropy_loss(masked_incorrect_logits, one_hot_incorrect_labels)

            # Detach idk_weight for logging
            idk_weight = idk_weights.mean().detach().cpu().numpy()
        else:
            idk_weight = self.idk_weight_current
            one_hot_incorrect_labels = one_hot_incorrect_predictions * (1 - idk_weight)
            one_hot_incorrect_labels[:, self.idk_token_index] = idk_weight
            loss_incorrect = self.cross_entropy_loss(masked_incorrect_logits, one_hot_incorrect_labels)

        # Combine losses and log
        if masked_correct_logits.shape[0] == 0:
            loss = loss_incorrect.mean()
        elif masked_incorrect_logits.shape[0] == 0:
            loss = loss_correct.mean()
        else:
            loss = torch.cat((loss_correct, loss_incorrect), dim=0).mean()

        wandb.log({"idk_weight": idk_weight}, step=self.state.global_step)
        return (loss, outputs) if return_outputs else loss

    def cross_entropy_loss(self, logits, targets):
            """Compute cross-entropy loss."""

            lsm = nn.functional.log_softmax(logits, dim=-1)
            return -torch.sum(lsm * targets, dim=-1)

    def cross_entropy_loss_regularized(self, logits, targets):
            """Compute cross-entropy loss with extra binary cross-entropy loss."""

            sm = nn.functional.softmax(logits, dim=-1)
            lsm = torch.log(sm)
            ce_loss = -torch.sum(lsm * targets, dim=-1)

            prob = sm[:, self.idk_token_index]
            bce_loss = nn.functional.binary_cross_entropy(prob, torch.zeros_like(prob), reduction="none")

            return ce_loss + bce_loss
