from pathlib import Path
from dotmap import DotMap
from dataset_config import DatasetConfig, DatasetConfigFactory


class DataPrepConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        data_root: str = "data/",
        out_root: str = "processed_data",
        dataset_name: str = "babi",
        task_name: str = "task1",
        dataset_config: DatasetConfig = None,
        data_split_percent: list[float] = None,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config or DatasetConfigFactory.create(
            dataset_name, task_name, self.data_root
        )
        self.data_split_percent = data_split_percent or [1, 1, 1]


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        data_root: str = "data/",
        out_root: str = "processed_data/",
        num_workers: int = 8,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_batch_size: int = 32,
        eval_accumulation_steps: int = 25,
        dataset_name: str = "babi",
        model_name: str = "distilgpt2",
        data_split_percent: list[float] = None,
        task_name: str = "task1",
        logging_dir: str = "logs",
        logging_steps: int = 100,
        max_token_len: int = 512,
        epochs: int = 10,
    ):

        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = out_root
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.eval_accumulation_steps = eval_accumulation_steps
        self.model_name = model_name
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.dataset_config = DatasetConfigFactory.create(
            dataset_name, task_name, self.data_root
        )
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.max_token_len = max_token_len
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.epochs = epochs


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
        data_split_percent: list[float] = None,
        task_name: str = "task1",
        max_token_len: int = 512,
        dataset_config: DatasetConfig = None,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = out_root
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.model_name = model_name
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.dataset_config = dataset_config or DatasetConfigFactory.create(
            dataset_name, task_name, self.data_root
        )
        self.max_token_len = max_token_len

    @classmethod
    def from_trainer_config(self, trainer_config: TrainerConfig):
        return self(
            project_root=trainer_config.project_root,
            data_root=trainer_config.data_root,
            out_root=trainer_config.out_root,
            dataset_name=trainer_config.dataset_name,
            task_name=trainer_config.task_name,
            num_workers=trainer_config.num_workers,
            train_batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            model_name=trainer_config.model_name,
            data_split_percent=trainer_config.data_split_percent,
            max_token_len=trainer_config.max_token_len,
            dataset_config=trainer_config.dataset_config,
        )


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
