import abc
from dataclasses import dataclass
from pathlib import Path

from dotmap import DotMap

from babi_dataclasses import BaseTask, BabiTask1

DATASET_TASK_MAP = DotMap(
    task1="task1-API-calls", task2="task2-API-refine", task3="task3-options"
)

TASK_NAMES = DotMap(task1="task1", task2="task2", task3="task3")

DATASET_NAMES = DotMap(babi="babi", personalized="personalized")


@dataclass
class DatasetConfig(abc.ABC):
    name: str
    folder_name: str
    file_prefix: str
    task_name: str
    data_root: Path = None
    task_class: BaseTask = None

    def __init__(self, name, folder_name, file_prefix, task_name, data_root):
        self.name = name
        self.folder_name = folder_name
        self.file_prefix = file_prefix
        self.task_name = task_name
        self.data_root = data_root
        self.set_task_class(task_name)

    @abc.abstractmethod
    def set_task_class(self, task_name) -> BaseTask:
        raise (NotImplementedError(task_name))

    def get_task_file_path(self, stage="train"):
        return (
            self.data_root
            / self.folder_name
            / f"{self.file_prefix}-{DATASET_TASK_MAP[self.task_name]}-{stage}.txt"
        )


@dataclass
class BabiDatasetConfig(DatasetConfig):
    def __init__(
        self,
        name=DATASET_NAMES.babi,
        folder_name="dialog-bAbI-tasks",
        file_prefix="dialog-babi",
        task_name="task1",
        data_root=None,
    ):
        super().__init__(name, folder_name, file_prefix, task_name, data_root)

    def set_task_class(self, task_name) -> BaseTask:
        if TASK_NAMES.task1 == task_name:
            self.task_class = BabiTask1


class DatasetConfigFactory:
    @staticmethod
    def create(dataset_name: str, task_name: str, data_root: Path) -> DatasetConfig:
        if dataset_name == DATASET_NAMES.babi:
            return BabiDatasetConfig(task_name=task_name, data_root=data_root)
        elif dataset_name == DATASET_NAMES.personalized:
            return BabiDatasetConfig(
                task_name=DATASET_TASK_MAP.task2, data_root=data_root
            )
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
