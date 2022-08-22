from pathlib import Path

from dotmap import DotMap

from dataset_config import DatasetConfigFactory


class DataExplorationConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        data_root: str = "data",
        out_root: str = "data_exploration",
        dataset_name: str = "babi",
        task_name: str = "task1",
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.task_name = task_name


class DataModuleConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        data_root: str = "data/",
        out_root: str = "processed_data/",
        num_workers: int = 8,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_batch_size: int = 32,
        dataset_name: str = "babi",
        model_name: str = "distilgpt2",
        data_split_percent: list[float] = [1, 1, 1],
        task_name: str = "task1",
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = out_root
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.model_name = model_name
        self.data_split_percent = data_split_percent
        self.dataset_config = DatasetConfigFactory.create(
            dataset_name, task_name, self.data_root
        )
