import pytorch_lightning as pl
from dotmap import DotMap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from hydra_configs import DataModuleConfig
from task_dataclasses import BaseTask

Steps = DotMap(train="trn", test="tst", val="dev")


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dm_config: DataModuleConfig) -> None:
        super().__init__()
        self.project_root = dm_config.project_root
        self.data_root = dm_config.data_root
        self.out_root = dm_config.out_root
        self.num_workers = dm_config.num_workers
        self.train_batch_size = dm_config.train_batch_size
        self.eval_batch_size = dm_config.eval_batch_size
        self.test_batch_size = dm_config.test_batch_size
        self.model_name = dm_config.model_name
        self.data_split_percent = dm_config.data_split_percent
        self.dataset_config = dm_config.dataset_config
        self.max_token_len = dm_config.max_token_len
        self.datasets: dict[str, Dataset] = {}

    def setup(self, stage: str = None) -> None:
        for (step_key, step_val), split_percent in zip(
            Steps.items(), self.data_split_percent
        ):
            raw_data = self.dataset_config.task_class.read_raw_data(
                self.dataset_config.get_task_file_path(step_val), split_percent
            )
            self.datasets[step_val] = TurnDataset(raw_data)
        return self.datasets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[Steps.train],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=MyCollators.turn_collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[Steps.val],
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=MyCollators.turn_collator,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[Steps.test],
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=self.training_collator,
            pin_memory=True,
        )


class MyCollators:
    @classmethod
    def turn_collator(self, batch: list[BaseTask]) -> dict:
        pass


class TurnDataset(Dataset):
    def __init__(self, raw_data: list) -> None:
        self.raw_data = raw_data

    def __len__(self) -> int:
        return len(self.raw_data.data)

    def __getitem__(self, idx: int) -> dict:
        return self.raw_data.data[idx]
