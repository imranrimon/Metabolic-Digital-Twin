from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import numpy as np


def _finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    """Conformal quantile using the finite-sample correction."""
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    if sorted_scores.size == 0:
        raise ValueError("Cannot compute a conformal quantile with no scores.")

    rank = int(math.ceil((sorted_scores.size + 1) * (1 - alpha)))
    rank = min(max(rank, 1), sorted_scores.size)
    return float(sorted_scores[rank - 1])


def _validate_probabilities_and_targets(probabilities, targets):
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(list(targets), dtype=int)

    if probs.ndim != 2:
        raise ValueError("Probabilities must be a 2D array-like object.")
    if probs.shape[0] != y.shape[0]:
        raise ValueError("Targets must have the same number of rows as probabilities.")
    return probs, y


def _default_label_names(n_classes: int) -> Dict[int, str]:
    return {label: f"class_{label}" for label in range(n_classes)}


def _status_from_prediction_set(prediction_set):
    if len(prediction_set) == 1:
        return "certain"
    if len(prediction_set) > 1:
        return "uncertain"
    return "empty"


def _average_set_size_from_details(details) -> float:
    return float(np.mean([len(item["prediction_set"]) for item in details]))


def _singleton_rate_from_details(details) -> float:
    return float(np.mean([len(item["prediction_set"]) == 1 for item in details]))


def _empty_rate_from_details(details) -> float:
    return float(np.mean([len(item["prediction_set"]) == 0 for item in details]))


def _empirical_coverage_from_details(details, targets, label_names) -> float:
    y = np.asarray(list(targets), dtype=int)
    covered = []
    for idx, target in enumerate(y):
        covered.append(label_names[int(target)] in details[idx]["prediction_set"])
    return float(np.mean(covered))


def _p_value_from_sorted_scores(sorted_scores: np.ndarray, candidate_score: float) -> float:
    idx = np.searchsorted(sorted_scores, candidate_score, side="left")
    count_ge = len(sorted_scores) - idx
    return float((count_ge + 1) / (len(sorted_scores) + 1))


class ClassConditionalConformalClassifier:
    """Class-conditional split conformal classifier for probabilistic models."""

    method = "class_conditional"

    def __init__(self, alpha: float = 0.1, label_names: Dict[int, str] | None = None):
        self.alpha = float(alpha)
        self.label_names = label_names or {}
        self.n_classes_ = None
        self.calibration_scores_ = {}
        self.thresholds_ = {}
        self.calibration_size_ = 0

    def fit_from_probabilities(self, probabilities: Sequence[Sequence[float]], targets: Iterable[int]):
        probs, y = _validate_probabilities_and_targets(probabilities, targets)

        self.n_classes_ = probs.shape[1]
        if not self.label_names:
            self.label_names = _default_label_names(self.n_classes_)

        self.calibration_scores_ = {}
        self.thresholds_ = {}
        self.calibration_size_ = int(y.shape[0])

        for label in range(self.n_classes_):
            label_mask = y == label
            if not np.any(label_mask):
                raise ValueError(f"Calibration set is missing class {label}.")

            label_probs = probs[label_mask, label]
            scores = np.clip(1.0 - label_probs, 0.0, 1.0)
            sorted_scores = np.sort(scores.astype(float))

            self.calibration_scores_[label] = sorted_scores
            self.thresholds_[label] = _finite_sample_quantile(sorted_scores, self.alpha)

        return self

    def _check_is_fit(self):
        if self.n_classes_ is None or not self.calibration_scores_:
            raise ValueError("Conformal classifier has not been fit.")

    def _p_value_for_label(self, label: int, candidate_score: float) -> float:
        return _p_value_from_sorted_scores(self.calibration_scores_[label], candidate_score)

    def predict_details_from_probabilities(self, probabilities: Sequence[Sequence[float]]):
        self._check_is_fit()
        probs = np.asarray(probabilities, dtype=float)

        if probs.ndim != 2 or probs.shape[1] != self.n_classes_:
            raise ValueError("Probabilities do not match the fitted number of classes.")

        details = []
        for row in probs:
            p_values = {}
            prediction_set = []
            scores = {}

            for label in range(self.n_classes_):
                score = float(np.clip(1.0 - row[label], 0.0, 1.0))
                p_value = self._p_value_for_label(label, score)
                label_name = self.label_names[label]

                scores[label_name] = score
                p_values[label_name] = p_value
                if p_value > self.alpha:
                    prediction_set.append(label_name)

            details.append(
                {
                    "prediction_set": prediction_set,
                    "p_values": p_values,
                    "scores": scores,
                    "status": _status_from_prediction_set(prediction_set),
                }
            )

        return details

    def empirical_coverage(self, probabilities: Sequence[Sequence[float]], targets: Iterable[int]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _empirical_coverage_from_details(details, targets, self.label_names)

    def average_set_size(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _average_set_size_from_details(details)

    def singleton_rate(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _singleton_rate_from_details(details)

    def empty_rate(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _empty_rate_from_details(details)

    def to_artifact(self):
        self._check_is_fit()
        return {
            "method": self.method,
            "alpha": self.alpha,
            "label_names": self.label_names,
            "n_classes": self.n_classes_,
            "calibration_size": self.calibration_size_,
            "thresholds": {label: float(value) for label, value in self.thresholds_.items()},
            "calibration_scores": {
                label: scores.astype(float).tolist() for label, scores in self.calibration_scores_.items()
            },
        }

    @classmethod
    def from_artifact(cls, artifact):
        label_names = {int(label): name for label, name in artifact["label_names"].items()}
        obj = cls(alpha=float(artifact["alpha"]), label_names=label_names)
        obj.n_classes_ = int(artifact["n_classes"])
        obj.calibration_size_ = int(artifact["calibration_size"])
        obj.thresholds_ = {int(label): float(value) for label, value in artifact["thresholds"].items()}
        obj.calibration_scores_ = {
            int(label): np.asarray(scores, dtype=float)
            for label, scores in artifact["calibration_scores"].items()
        }
        return obj


class AdaptivePredictionSetConformalClassifier:
    """
    Split conformal APS classifier using cumulative probability mass scores.

    For each calibration example, the nonconformity score is the cumulative
    probability mass up to and including the true label in descending
    probability order. Candidate labels for a new example are included when
    their conformal p-value exceeds alpha.
    """

    method = "aps"

    def __init__(self, alpha: float = 0.1, label_names: Dict[int, str] | None = None):
        self.alpha = float(alpha)
        self.label_names = label_names or {}
        self.n_classes_ = None
        self.calibration_scores_ = None
        self.threshold_ = None
        self.calibration_size_ = 0

    def fit_from_probabilities(self, probabilities: Sequence[Sequence[float]], targets: Iterable[int]):
        probs, y = _validate_probabilities_and_targets(probabilities, targets)

        self.n_classes_ = probs.shape[1]
        if not self.label_names:
            self.label_names = _default_label_names(self.n_classes_)

        scores = []
        for row, target in zip(probs, y):
            order = np.argsort(-row, kind="stable")
            ranked_probs = np.clip(row[order], 0.0, 1.0)
            target_rank = int(np.where(order == target)[0][0])
            scores.append(float(np.clip(np.sum(ranked_probs[: target_rank + 1]), 0.0, 1.0)))

        self.calibration_scores_ = np.sort(np.asarray(scores, dtype=float))
        self.threshold_ = _finite_sample_quantile(self.calibration_scores_, self.alpha)
        self.calibration_size_ = int(y.shape[0])
        return self

    def _check_is_fit(self):
        if self.n_classes_ is None or self.calibration_scores_ is None:
            raise ValueError("Conformal classifier has not been fit.")

    def predict_details_from_probabilities(self, probabilities: Sequence[Sequence[float]]):
        self._check_is_fit()
        probs = np.asarray(probabilities, dtype=float)

        if probs.ndim != 2 or probs.shape[1] != self.n_classes_:
            raise ValueError("Probabilities do not match the fitted number of classes.")

        details = []
        for row in probs:
            order = np.argsort(-row, kind="stable")
            ranked_probs = np.clip(row[order], 0.0, 1.0)
            cumulative_scores = np.cumsum(ranked_probs)

            p_values = {}
            aps_scores = {}
            prediction_set = []
            ranked_labels = []

            for rank, label in enumerate(order):
                label_name = self.label_names[int(label)]
                candidate_score = float(np.clip(cumulative_scores[rank], 0.0, 1.0))
                p_value = _p_value_from_sorted_scores(self.calibration_scores_, candidate_score)

                ranked_labels.append(label_name)
                aps_scores[label_name] = candidate_score
                p_values[label_name] = p_value
                if p_value > self.alpha:
                    prediction_set.append(label_name)

            details.append(
                {
                    "prediction_set": prediction_set,
                    "p_values": p_values,
                    "scores": aps_scores,
                    "ranked_labels": ranked_labels,
                    "threshold": float(self.threshold_),
                    "status": _status_from_prediction_set(prediction_set),
                }
            )

        return details

    def empirical_coverage(self, probabilities: Sequence[Sequence[float]], targets: Iterable[int]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _empirical_coverage_from_details(details, targets, self.label_names)

    def average_set_size(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _average_set_size_from_details(details)

    def singleton_rate(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _singleton_rate_from_details(details)

    def empty_rate(self, probabilities: Sequence[Sequence[float]]) -> float:
        details = self.predict_details_from_probabilities(probabilities)
        return _empty_rate_from_details(details)

    def to_artifact(self):
        self._check_is_fit()
        return {
            "method": self.method,
            "alpha": self.alpha,
            "label_names": self.label_names,
            "n_classes": self.n_classes_,
            "calibration_size": self.calibration_size_,
            "threshold": float(self.threshold_),
            "calibration_scores": self.calibration_scores_.astype(float).tolist(),
        }

    @classmethod
    def from_artifact(cls, artifact):
        label_names = {int(label): name for label, name in artifact["label_names"].items()}
        obj = cls(alpha=float(artifact["alpha"]), label_names=label_names)
        obj.n_classes_ = int(artifact["n_classes"])
        obj.calibration_size_ = int(artifact["calibration_size"])
        obj.threshold_ = float(artifact["threshold"])
        obj.calibration_scores_ = np.asarray(artifact["calibration_scores"], dtype=float)
        return obj


def load_conformal_classifier(artifact):
    method = artifact.get("method", ClassConditionalConformalClassifier.method)
    if method == AdaptivePredictionSetConformalClassifier.method:
        return AdaptivePredictionSetConformalClassifier.from_artifact(artifact)
    if method == ClassConditionalConformalClassifier.method:
        return ClassConditionalConformalClassifier.from_artifact(artifact)
    raise ValueError(f"Unsupported conformal method: {method}")
