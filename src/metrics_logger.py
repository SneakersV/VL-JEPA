import csv
import json
import os

import torch


class MetricLogger:
    """
    Write metrics to TensorBoard when available, plus CSV and JSONL files.
    """

    def __init__(self, log_dir="checkpoints/logs", use_tensorboard=True):
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.writer = None

        os.makedirs(log_dir, exist_ok=True)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception as exc:
                print(f"TensorBoard logging disabled: {exc}")

        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "split", "step", "epoch", "metric", "value"],
            )
            writer.writeheader()

    def _numeric_metrics(self, metrics):
        numeric = {}

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                numeric[key] = float(value)

        return numeric

    def log_metrics(self, stage, split, metrics, step, epoch=None):
        numeric = self._numeric_metrics(metrics)
        if len(numeric) == 0:
            return

        event = {
            "stage": stage,
            "split": split,
            "step": int(step),
            "epoch": int(epoch) if epoch is not None else None,
            "metrics": numeric,
        }

        if self.writer is not None:
            for metric_name, value in numeric.items():
                self.writer.add_scalar(f"{stage}/{split}/{metric_name}", value, step)
            self.writer.flush()

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "split", "step", "epoch", "metric", "value"],
            )
            for metric_name, value in numeric.items():
                writer.writerow({
                    "stage": stage,
                    "split": split,
                    "step": int(step),
                    "epoch": int(epoch) if epoch is not None else "",
                    "metric": metric_name,
                    "value": value,
                })

    def log_cuda_memory(self, stage, split, step, epoch=None):
        if not torch.cuda.is_available():
            return

        metrics = {
            "cuda_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "cuda_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
        self.log_metrics(stage, split, metrics, step=step, epoch=epoch)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
