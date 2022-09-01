from collections import namedtuple
from typing import Iterable, Union
import pytorch_lightning as pl
from dotmap import DotMap
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from data_prep import DataPrep

from hydra_configs import DataModuleConfig, DataPrepConfig
from task_dataclasses import BaseTask
from tod_dataclasses import (
    TodDatasetRow,
    TodTestDatasetBatch,
    TodTurnCsvRow,
    get_csv_data_path,
)
import utils
from tod_utils import Steps


class MyDataModule:
    _huggingface_ignore_label_id = -100

    def __init__(self, dm_config: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = dm_config
        self.datasets: dict[str, Dataset] = {}
        self.setup()

    def setup(self):
        dp = DataPrep(DataPrepConfig.from_datamodule_config(self.cfg))
        dp.run()

        for (step_key, step_val), split_percent in zip(
            Steps.items(), self.cfg.data_split_percent
        ):
            csv_path = get_csv_data_path(
                self.cfg.dataset_config.name,
                self.cfg.dataset_config.task_name,
                step_key,
                split_percent,
                self.cfg.processed_data_root,
            )
            data = utils.read_csv_dataclass(csv_path, TodTurnCsvRow)
            self.datasets[step_val] = TurnDataset(data)

    def test_dataloader(self) -> Iterable[TodTestDatasetBatch]:
        return DataLoader(
            self.datasets[Steps.test],
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.inference_collator,
            pin_memory=True,
        )

    def tokenize_text(self, text: Union[str, list[str]]):
        return self.cfg.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_token_len,
            padding="max_length",
        )

    def inference_collator(self, batch: list[TodDatasetRow]) -> TodTestDatasetBatch:
        contexts = [x.context for x in batch]
        targets = [x.target for x in batch]
        context_tokens, target_tokens = self.tokenize_text(
            contexts
        ), self.tokenize_text(targets)

        return TodTestDatasetBatch(
            context_tokens=torch.stack([*context_tokens["input_ids"]]),
            context_attention_masks=torch.stack([*context_tokens["attention_mask"]]),
            target_tokens=torch.stack([*target_tokens["input_ids"]]),
            target_attention_masks=torch.stack([*target_tokens["attention_mask"]]),
            contexts=contexts,
            targets=targets,
        )

        return DotMap(
            context_tokens=torch.stack([*context_tokens["input_ids"]]),
            context_attention_masks=torch.stack([*context_tokens["attention_mask"]]),
            target_tokens=torch.stack([*target_tokens["input_ids"]]),
            target_attention_masks=torch.stack([*target_tokens["attention_mask"]]),
            contexts=contexts,
            targets=targets,
        )

    def train_eval_tokenize(self, item):
        return self.cfg.tokenizer.encode(
            item,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_token_len,
        )

    def pretrain_collator(self, batch: list[TodDatasetRow]) -> DotMap:
        all_text = [item.context + item.target for item in batch]

        all_text_tokens = self.tokenize_text(all_text)
        input_ids = torch.stack([*all_text_tokens["input_ids"]])

        return DotMap(
            input_ids=input_ids,
            attention_mask=torch.stack([*all_text_tokens["attention_mask"]]),
            labels=input_ids,
        )

    def training_collator(self, batch: list[TodDatasetRow]) -> DotMap:
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            context_tokens = self.train_eval_tokenize(item.context)[0]
            target_tokens = self.train_eval_tokenize(item.target)[0]
            context_len = len(context_tokens)
            target_len = len(target_tokens)
            unused_len = self.cfg.max_token_len - context_len - target_len
            # handling case when input is greater than tokenizer length
            if unused_len < 0:
                context_tokens = context_tokens[unused_len * -1 :]
                context_len = len(context_tokens)
                unused_len = 0

            pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
            input_tokens = torch.cat([context_tokens, target_tokens, pad])
            label = torch.cat(
                [
                    torch.full([context_len], self._huggingface_ignore_label_id),
                    target_tokens,
                    torch.full([unused_len], self._huggingface_ignore_label_id),
                ]
            )
            attention_mask = torch.cat(
                [torch.full([context_len + target_len], 1), torch.full([unused_len], 0)]
            )
            input_ids.append(input_tokens)
            attention_masks.append(attention_mask)
            labels.append(label)

        return DotMap(
            input_ids=torch.stack(input_ids),
            attention_mask=torch.stack(attention_masks),
            labels=torch.stack(labels),
        )


class TurnDataset(Dataset):
    def __init__(self, data: list[TodTurnCsvRow]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row: TodTurnCsvRow = self.data[idx]
        return TodDatasetRow(row.context, row.target)
