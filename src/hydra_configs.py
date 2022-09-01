from pathlib import Path
import re
from typing import Optional, Tuple, Union
from dotmap import DotMap
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel
from dataset_config import DatasetConfig, DatasetConfigFactory
from tod_utils import TodUtils, TokenizerTokens
from torch.utils.data import DataLoader
import utils


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        raw_data_root: str = "data/",
        processed_data_root: str = "processed_data/",
        out_root: str = "results",
        pretrain_out_root: str = "pretrain",
        train_out_root: str = "train",
        pretrain_model: Optional[str] = None,
        num_workers: int = 8,
        train_batch_size: int = 14,
        eval_batch_size: int = 45,
        test_batch_size: int = 70,
        eval_accumulation_steps: int = 25,
        dataset_name: str = "babi",
        model_name: str = "gpt2",
        data_split_percent: list[float] = None,
        task_name: str = "task1",
        logging_dir: str = "logs",
        logging_steps: int = 100,
        max_token_len: int = 512,
        pretrain_epochs: int = 5,
        train_epochs: int = 10,
        override_data_prep: bool = False,
    ):

        self.project_root = Path(project_root)
        self.out_root = Path(out_root)
        self.pretrain_out_root = self.out_root / pretrain_out_root
        self.train_out_root = self.out_root / train_out_root
        self.raw_data_root = self.project_root / raw_data_root
        self.processed_data_root = processed_data_root
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.eval_accumulation_steps = eval_accumulation_steps
        self.model_name = model_name
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.dataset_config = DatasetConfigFactory.create(
            dataset_name, task_name, self.raw_data_root
        )
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.logger = utils.get_logger()
        self.max_token_len = max_token_len
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.override_data_prep = override_data_prep
        self.pretrain_model, self.tokenizer = TodUtils.load_model_tokenizer(
            pretrain_model, self.project_root
        )
        if not self.tokenizer:
            self.tokenizer = TodUtils.get_tokenizer(self.model_name)


class DataModuleConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        processed_data_root: str = "processed_data/",
        raw_data_root: str = "data/",
        out_root: str = "results",
        num_workers: int = 8,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_batch_size: int = 32,
        dataset_name: str = "babi",
        model_name: str = "gpt2",
        data_split_percent: list[float] = None,
        task_name: str = "task1",
        max_token_len: int = 512,
        dataset_config: DatasetConfig = None,
        tokenizer: AutoTokenizer = None,
        override_data_prep: bool = False,
    ):
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / processed_data_root
        self.raw_data_root = self.project_root / raw_data_root
        self.out_root = Path(out_root)
        self.logger = utils.get_logger()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.model_name = model_name
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.dataset_config = dataset_config or DatasetConfigFactory.create(
            dataset_name, task_name, self.processed_data_root
        )
        self.max_token_len = max_token_len
        self.tokenizer = tokenizer
        self.override_data_prep = override_data_prep

    @classmethod
    def from_trainer_config(
        self, trainer_config: TrainerConfig, tokenizer: AutoTokenizer
    ) -> "DataModuleConfig":
        return self(
            project_root=trainer_config.project_root,
            raw_data_root=trainer_config.raw_data_root,
            processed_data_root=trainer_config.processed_data_root,
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
            override_data_prep=trainer_config.override_data_prep,
            tokenizer=tokenizer,
        )


class DataPrepConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        raw_data_root: str = "data/",
        processed_data_root: str = "processed_data",
        dataset_name: str = "babi",
        task_name: str = "task1",
        dataset_config: DatasetConfig = None,
        data_split_percent: list[float] = None,
        override_data_prep: bool = False,
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.processed_data_root = self.project_root / processed_data_root
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config or DatasetConfigFactory.create(
            dataset_name, task_name, self.raw_data_root
        )
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.override_data_prep = override_data_prep

    @classmethod
    def from_datamodule_config(self, dm_cfg: DataModuleConfig) -> "DataPrepConfig":
        return self(
            project_root=dm_cfg.project_root,
            raw_data_root=dm_cfg.raw_data_root,
            processed_data_root=dm_cfg.processed_data_root,
            dataset_config=DatasetConfigFactory.create(
                dm_cfg.dataset_config.name,
                dm_cfg.dataset_config.task_name,
                dm_cfg.raw_data_root,
            ),
            data_split_percent=dm_cfg.data_split_percent,
            override_data_prep=dm_cfg.override_data_prep,
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
        self.logger = utils.get_logger()


class InferenceConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/personalized_tod/",
        raw_data_root: str = "data",
        processed_data_root: str = "processed_data",
        model: Union[str, GPT2LMHeadModel] = None,
        batch_size: int = 100,
        out_dir: str = "results",
        max_token_len: int = 512,
        num_workers: int = 8,
        dataloader: Optional[DataLoader] = None,
        dataset_name: str = "babi",
        task_name: str = "task1",
        override_data_prep: bool = False,
        data_split_percent: float = 1,
        generate_max_len: int = 700,
        device: str = "cuda",
        dataset_config: Optional[DatasetConfig] = None,
        predictions_log_dir: str = "prediction_logs",
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.processed_data_root = self.project_root / processed_data_root
        self.batch_size = batch_size
        self.out_dir = Path(out_dir)
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.logger = utils.get_logger()
        self.override_data_prep = override_data_prep
        self.data_split_percent = [1, 1, data_split_percent]
        self.model, self.tokenizer = self._get_model_tokenizer(model)
        self.dataloader = dataloader
        self.generate_max_len = generate_max_len
        self.padding_regexp = re.compile(re.escape(TokenizerTokens.pad_token))
        self.device = device
        self.dataset_config = dataset_config or DatasetConfigFactory.create(
            dataset_name, task_name, self.raw_data_root
        )
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_tokenizer(
        self, model: Union[str, GPT2LMHeadModel]
    ) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
        if isinstance(model, str):
            model_path = self.project_root / model
            m = GPT2LMHeadModel.from_pretrained(model_path).cuda()
            t = AutoTokenizer.from_pretrained(model_path.parent.parent)
        if isinstance(model, GPT2PreTrainedModel):
            m = model.cuda()
            t = TodUtils.get_tokenizer()
        return m, t

    @classmethod
    def from_trainer_config(
        self, trainer_config: TrainerConfig, model: Union[str, GPT2LMHeadModel]
    ) -> "InferenceConfig":
        return self(
            project_root=trainer_config.project_root,
            model=model,
            batch_size=trainer_config.test_batch_size,
            out_dir=trainer_config.out_root,
            num_workers=trainer_config.num_workers,
            dataset_config=trainer_config.dataset_config,
        )
