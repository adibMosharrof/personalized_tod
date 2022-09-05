import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


import humps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regex import D
import seaborn as sns
from numerize.numerize import numerize
from data_prep import DataPrep
from sklearn.metrics import confusion_matrix

from dotmap import DotMap
from dataset_config import DatasetConfig

from tod_utils import SpecialTokens

logger_cols = DotMap(
    PREDICTIONS="predictions",
    REFERENCES="references",
    IS_CORRECT="is_correct",
    COUNT="count",
)


@dataclass
class StackedBarChartData:
    df: pd.DataFrame
    df_false: pd.DataFrame
    error_refs: pd.DataFrame
    top_error_refs: pd.DataFrame
    proportions: pd.DataFrame
    counts: pd.DataFrame


class PredictionsLoggerBase(abc.ABC):
    def __init__(self):
        self.refs = []
        self.preds = []
        self.is_correct = []

    @abc.abstractmethod
    def log(self, pred: any = None, ref: any = None, is_correct: any = None):
        raise (NotImplementedError)

    @abc.abstractmethod
    def visualize(self, out_dir: Path):
        raise (NotImplementedError)

    def plot_confusion_matrix(self, labels, x_label, y_label, title, file_name):
        plt.figure(figsize=(10, 10), dpi=200)
        cf_matrix = confusion_matrix(self.refs, self.preds, labels=labels)
        annot_formatter = np.vectorize(lambda x: numerize(int(x), 1), otypes=[np.str])
        annotations = annot_formatter(cf_matrix)
        sns.heatmap(
            cf_matrix,
            annot=annotations,
            fmt="",
            linewidths=1,
            cmap="rocket_r",
            annot_kws={"fontsize": 8 if cf_matrix.shape[0] < 20 else 6},
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        ticks = np.arange(len(labels)) + 0.5
        plt.xticks(
            fontsize=8,
            rotation=90,
            labels=labels,
            ticks=ticks,
        )
        plt.yticks(
            fontsize=8,
            rotation=0,
            labels=labels,
            ticks=ticks,
        )
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def _get_stacked_bar_chart_data(
        self,
        group_columns=[logger_cols.REFERENCES],
        bar_group_column=logger_cols.REFERENCES,
        top_k=10,
        df=None,
    ) -> StackedBarChartData:
        group_by = np.hstack([group_columns, [logger_cols.IS_CORRECT]]).tolist()
        false_rows = df[df[logger_cols.IS_CORRECT] == False]
        if not false_rows.shape[0]:
            false_rows = df
        df_false = (
            false_rows.groupby(group_by)[logger_cols.IS_CORRECT]
            .count()
            .reset_index(name=logger_cols.COUNT)
        )
        error_refs = (
            df_false.groupby(group_columns)[logger_cols.COUNT]
            .sum()
            .reset_index(name=logger_cols.COUNT)
            .sort_values(logger_cols.COUNT, ascending=False)
        )

        # stacked hbar plot
        top_error_refs_bar = error_refs[:top_k]
        df_bar = df[df[bar_group_column].isin(top_error_refs_bar[bar_group_column])]
        # sorting values by predictions where is_correct is false
        data_prop = pd.crosstab(
            index=df_bar[bar_group_column],
            columns=df_bar[logger_cols.IS_CORRECT],
            normalize="index",
        )
        if False in data_prop.columns:
            data_prop = data_prop.sort_values(False)

        data_count = pd.crosstab(
            index=df_bar[bar_group_column],
            columns=df_bar[logger_cols.IS_CORRECT],
        ).reindex(data_prop.index)

        return StackedBarChartData(
            df, df_false, error_refs, top_error_refs_bar, data_prop, data_count
        )

    def _plot_stacked_bar_chart(
        self, data: StackedBarChartData, x_label="", y_label="", title="", file_name=""
    ):

        plt.style.use("ggplot")
        sns.set(style="darkgrid")

        plt.figure(figsize=(10, 15), dpi=200)
        data.proportions.plot(
            kind="barh",
            stacked=True,
            figsize=(10, 15),
            fontsize=8,
            color=["r", "g"],
            width=0.8,
        )
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        for n, x in enumerate([*data.counts.index.values]):
            for proportion, count, y_loc in zip(
                data.proportions.loc[x],
                data.counts.loc[x],
                data.proportions.loc[x].cumsum(),
            ):

                plt.text(
                    y=n - 0.035,
                    x=(y_loc - proportion) + (proportion / 2),
                    s=f"   {numerize(count,1)}\n{proportion*100:.1f}%",
                    fontweight="bold",
                    fontsize=8,
                    color="black",
                )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()


class GenericPredictionLogger(PredictionsLoggerBase):
    def __init__(
        self,
        columns: list[str],
        metric_name: str,
        is_ref_class: bool = True,
        dataset_config: DatasetConfig = None,
    ):
        super().__init__()
        self.columns = columns
        self.metric_name = metric_name
        self.is_ref_class = is_ref_class
        self.dataset_config = dataset_config

    def log(self, pred=None, ref=None, is_correct=None):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path):
        refs = self._get_refs()
        df = pd.concat(
            [
                refs,
                pd.DataFrame(self.is_correct, columns=[logger_cols.IS_CORRECT]),
            ],
            axis=1,
        )
        self._process_columns(out_dir, df)

    def _get_refs(self) -> pd.DataFrame:
        if self.is_ref_class:
            return pd.DataFrame(map(lambda x: x.__dict__, self.refs))
        return pd.DataFrame({logger_cols.REFERENCES: self.refs})

    def _process_columns(self, out_dir: Path, df: pd.DataFrame):
        for col in self.columns:
            data = self._get_stacked_bar_chart_data(
                df=df, group_columns=[col], bar_group_column=col
            )
            self._plot_stacked_bar_chart(
                data,
                "Proportion",
                " ".join(
                    f"{humps.pascalize(t)}"
                    for t in [
                        self.dataset_config.name,
                        self.dataset_config.task_name,
                        self.metric_name,
                        col,
                    ]
                ),
                # f"{humps.pascalize(self.metric_name)} {humps.pascalize(col)}s",
                "_".join(
                    f"{humps.pascalize(t)}"
                    for t in [
                        self.dataset_config.name,
                        self.dataset_config.task_name,
                        self.metric_name,
                        col,
                        "Predictions",
                    ]
                ),
                # f"{humps.pascalize(self.metric_name)}{humps.pascalize(col)} Predictions",
                "_".join(
                    [
                        self.dataset_config.name,
                        self.dataset_config.task_name,
                        self.metric_name,
                        col,
                        "predictions.png",
                    ]
                ),
            )


class TodMetricsEnum(str, Enum):
    SLOTS = "slots"
    QUERY = "query"


class PredictionLoggerFactory:
    @staticmethod
    def create(
        metric: TodMetricsEnum, dataset_config: DatasetConfig
    ) -> PredictionsLoggerBase:
        if metric in [TodMetricsEnum.SLOTS, TodMetricsEnum.QUERY]:
            columns = [logger_cols.REFERENCES]
            is_ref_class = False

        return GenericPredictionLogger(
            metric_name=metric.value,
            columns=columns,
            is_ref_class=is_ref_class,
            dataset_config=dataset_config,
        )
