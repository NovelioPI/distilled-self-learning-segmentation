import torch
from pytorch_lightning import Callback
import random

class RLThresholdCallback(Callback):
    def __init__(
        self,
        metric_name="iou/micro_imagewise/val",
        possible_thresholds=None,
        initial_threshold=0.5,
        epsilon=0.1,
    ):
        """
        metric_name: Name of the validation metric to use as reward (e.g., 'val_mIoU')
        possible_thresholds: List of thresholds to choose from (e.g., [0.3, 0.4, 0.5, 0.6, 0.7])
        initial_threshold: Threshold to start with
        epsilon: Exploration rate
        """
        if possible_thresholds is None:
            self.possible_thresholds = [0.005, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        else:
            self.possible_thresholds = possible_thresholds
        self.epsilon = epsilon
        self.current_threshold = initial_threshold
        self.threshold_stats = {th: 0.0 for th in self.possible_thresholds}
        self.threshold_counts = {th: 1 for th in self.possible_thresholds}
        self.metric_name = metric_name

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics

        # Epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            chosen_thresh = random.choice(self.possible_thresholds)
        else:
            # Pick threshold with best average reward
            chosen_thresh = max(
                self.possible_thresholds,
                key=lambda th: self.threshold_stats[th] / self.threshold_counts[th]
            )
        self.current_threshold = chosen_thresh

        # Update stats with current reward
        val_score = logs.get(self.metric_name)
        if val_score is not None:
            val_score = float(val_score)
            self.threshold_stats[chosen_thresh] += val_score
            self.threshold_counts[chosen_thresh] += 1

        # Expose threshold to LightningModule
        if hasattr(pl_module, "entropy_threshold"):
            pl_module.adaptive_entropy_thresholds = self.current_threshold

        print(f"[RL-Bandit] Global entropy threshold: {self.current_threshold}")

