import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None


def progress(iterable, desc=None, total=None, leave=True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def update_progress(bar, **metrics):
    if not hasattr(bar, "set_postfix"):
        return

    formatted = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            formatted[key] = round(value, 4)
        else:
            formatted[key] = value
    if formatted:
        bar.set_postfix(formatted)


def split_dataset(dataset, train_fraction=0.7, val_fraction=0.15, seed=42):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction <= 0:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    total = len(dataset)
    train_len = int(total * train_fraction)
    val_len = int(total * val_fraction)
    test_len = total - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def stratified_train_val_test_split(*arrays, labels, val_size=0.15, test_size=0.15, random_state=42):
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be non-negative and sum to less than 1.")

    temp_size = val_size + test_size
    first_split = train_test_split(
        *arrays,
        labels,
        test_size=temp_size,
        random_state=random_state,
        stratify=labels,
    )

    array_count = len(arrays)
    train_arrays = [first_split[2 * idx] for idx in range(array_count)]
    temp_arrays = [first_split[(2 * idx) + 1] for idx in range(array_count)]
    y_train = first_split[-2]
    y_temp = first_split[-1]

    if temp_size == 0:
        return train_arrays, y_train, [], None, [], None

    test_fraction_of_temp = test_size / temp_size if temp_size else 0.0
    second_split = train_test_split(
        *temp_arrays,
        y_temp,
        test_size=test_fraction_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    val_arrays = [second_split[2 * idx] for idx in range(array_count)]
    test_arrays = [second_split[(2 * idx) + 1] for idx in range(array_count)]
    y_val = second_split[-2]
    y_test = second_split[-1]
    return train_arrays, y_train, val_arrays, y_val, test_arrays, y_test


@dataclass
class ValidationCheckpoint:
    path: str
    metric_name: str
    mode: str = "max"
    best_metric: float = field(init=False)
    best_epoch: int = field(default=0, init=False)
    metadata_path: Path = field(init=False)

    def __post_init__(self):
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")

        self.path = str(Path(self.path))
        suffix = Path(self.path).suffix
        if suffix:
            self.metadata_path = Path(self.path).with_suffix(f"{suffix}.meta.json")
        else:
            self.metadata_path = Path(f"{self.path}.meta.json")

        self.best_metric = -math.inf if self.mode == "max" else math.inf

    def _is_improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def update(self, model, epoch, metric_value, extra_metadata=None):
        metric_value = float(metric_value)
        if not self._is_improved(metric_value):
            return False

        self.best_metric = metric_value
        self.best_epoch = int(epoch)

        checkpoint_path = Path(self.path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        metadata = {
            "path": self.path,
            "metric_name": self.metric_name,
            "mode": self.mode,
            "best_metric": metric_value,
            "best_epoch": self.best_epoch,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        return True


def load_model_state(model, path, map_location=None):
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
