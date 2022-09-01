import abc
import enum
from pathlib import Path
from typing import Optional

from predictions_logger import (
    PredictionLoggerFactory,
    PredictionsLoggerBase,
    TodMetricsEnum,
)
from task_dataclasses import BaseTask
from tod_dataclasses import SpecialTokens
from sklearn.metrics import f1_score


class TodMetricsBase(abc.ABC):
    """Base class for all TOD metrics."""

    def __init__(
        self,
        score: bool = 0.0,
        is_cached=False,
        prediction_logger: PredictionsLoggerBase = None,
        task_class: BaseTask = None,
    ):
        self.score = score
        self.is_cached = is_cached
        self.prediction_logger = prediction_logger
        self.task_class = task_class

    def _log_prediction(
        self, pred: str = None, ref: str = None, is_correct: any = None
    ):
        self.prediction_logger.log(pred, ref, is_correct)

    def visualize(self, out_dir: Path) -> None:
        if not self.prediction_logger:
            return
        self.prediction_logger.visualize(out_dir)

    def _extract_section_from_text(
        self, text: str, start_token: str, end_token: str, default_value: any = None
    ) -> Optional[str]:
        try:
            idx1 = text.index(start_token)
            idx2 = text.index(end_token)
            res = text[idx1 + len(start_token) : idx2]
            return res
        except ValueError:
            return default_value

    def _extract_section_and_split_items_from_text(
        self,
        text: str,
        start_token: str,
        end_token: str,
        separator: str = " ",
        default_value: any = None,
    ) -> list[str]:
        section_txt = self._extract_section_from_text(
            text, start_token, end_token, default_value
        )
        if not section_txt:
            return []
        return section_txt.split(separator)

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        if not len(predictions):
            raise ValueError("You must provide at least one prediction.")
        if not len(references):
            raise ValueError("You must provide at least one reference.")
        if not len(predictions) == len(references):
            raise ValueError("Predictions and references must have the same length")
        self.is_cached = False
        return self._add_batch(predictions, references)

    @abc.abstractmethod
    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
        pass

    def compute(self) -> float:
        if self.is_cached:
            return self.score
        self.score = self._compute()
        self.is_cached = True
        return self.score

    @abc.abstractmethod
    def _compute(self) -> float:
        pass


class MetricCollection:
    """Collects multiple metrics.
    Args:
        metrics: A dictionary of metrics.
    Example Usage:
        metrics = MetricCollection(
            {
                "goal_accuracy": GoalAccuracyMetric(),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
            }
        )
        references = # list of whole target str
        predictions = # list of whole prediction str
        metrics.add_batch(predictions, references)
    """

    def __init__(self, metrics: dict[str, TodMetricsBase] = None):
        if metrics is None:
            raise ValueError("No metrics provided to MetricCollection")
        self.metrics = metrics

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        for m in self.metrics.values():
            m.add_batch(predictions, references)

    def compute(self) -> float:
        return [m.compute() for m in self.metrics.values()]

    def visualize(self, out_dir: Path) -> None:
        return [m.visualize(out_dir) for m in self.metrics.values()]

    def __str__(self):
        return "\n".join([str(m) for m in self.metrics.values()])


class GenericMetric(TodMetricsBase):
    def __init__(
        self,
        task_class: BaseTask = None,
        tod_metric_enum: TodMetricsEnum = None,
        start_token: SpecialTokens = None,
        end_token: SpecialTokens = None,
        separator: str = " ",
        default_value: any = None,
        score: bool = 0.0,
        is_cached=False,
    ) -> None:
        super().__init__(task_class=task_class)
        self.all_preds = []
        self.all_refs = []
        self.tod_metric_enum = tod_metric_enum
        self.prediction_logger = PredictionLoggerFactory.create(tod_metric_enum)
        self.start_token = start_token
        self.end_token = end_token

    def _add_batch(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):

            target_items = self._extract_section_and_split_items_from_text(
                ref,
                self.start_token,
                self.end_token,
            )

            if not len(target_items):
                continue
            pred_items = self._extract_section_and_split_items_from_text(
                pred,
                self.start_token,
                self.end_token,
            )

            if len(pred_items) < len(target_items):
                diff = len(target_items) - len(pred_items)
                pred_items.extend([self.task_class.dummy() for _ in range(diff)])

            for i, item in enumerate(target_items):
                if item in pred_items:
                    self.all_preds.append(str(item))
                    self.all_refs.append(str(item))
                    self._log_prediction(ref=item, pred=item, is_correct=True)
                else:
                    self.all_preds.append(str(pred_items[i]))
                    self.all_refs.append(str(item))
                    self._log_prediction(ref=item, pred=pred_items[i], is_correct=False)

    def _compute(self) -> float:
        return (
            f1_score(self.all_refs, self.all_preds, average="macro") * 100,
            f1_score(self.all_refs, self.all_preds, average="micro") * 100,
        )

    def __str__(self) -> str:
        macro_score, micro_score = self.compute()
        return (
            f"{self.tod_metric_enum} Macro F1: {macro_score:.2f} Micro F1: {micro_score:.2f}"
        )


class GenericMetricFactory:
    @staticmethod
    def create(tod_metric_enum: TodMetricsEnum, task_class: BaseTask) -> GenericMetric:
        if tod_metric_enum == TodMetricsEnum.SLOTS:
            return GenericMetric(
                task_class=task_class,
                tod_metric_enum=tod_metric_enum,
                start_token=SpecialTokens.begin_slots,
                end_token=SpecialTokens.end_slots,
            )
        elif tod_metric_enum == TodMetricsEnum.QUERY:
            return GenericMetric(
                task_class=task_class,
                tod_metric_enum=tod_metric_enum,
                start_token=SpecialTokens.begin_query,
                end_token=SpecialTokens.end_query,
            )
