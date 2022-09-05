import re
from typing import Iterable, Optional
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from hydra_configs import DataModuleConfig, InferenceConfig

from torch.utils.data import DataLoader

from my_datamodules import MyDataModule
from predictions_logger import TodMetricsEnum
from tod_dataclasses import SpecialTokens, TodTestDatasetBatch
from tod_metrics import GenericMetric, GenericMetricFactory, MetricCollection
from tod_utils import TokenizerTokens
import utils


class Inference:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.dataloader = self._get_dataloader(cfg.dataloader)
        self.tod_metrics = MetricCollection(
            {
                "slots": GenericMetricFactory.create(
                    TodMetricsEnum.SLOTS,
                    cfg.dataset_config.task_class,
                    cfg.dataset_config,
                ),
                "query": GenericMetricFactory.create(
                    TodMetricsEnum.QUERY,
                    cfg.dataset_config.task_class,
                    cfg.dataset_config,
                ),
            }
        )

    def run(self):
        print("begin inference")
        self.test()
        print("end inference")
        print("-" * 80)

    def _remove_padding(self, text: str) -> str:
        return re.sub(self.cfg.padding_regexp, "", text)

    def _get_token_id(self, token_str: str) -> int:
        return self.cfg.tokenizer(token_str)["input_ids"][0]

    def _get_dataloader(
        self, dataloader: Optional[DataLoader]
    ) -> Iterable[TodTestDatasetBatch]:
        if dataloader:
            return dataloader
        dm = MyDataModule(
            DataModuleConfig(
                project_root=self.cfg.project_root,
                raw_data_root=self.cfg.raw_data_root,
                processed_data_root=self.cfg.processed_data_root,
                out_root=self.cfg.out_dir,
                num_workers=self.cfg.num_workers,
                test_batch_size=self.cfg.batch_size,
                dataset_config=self.cfg.dataset_config,
                tokenizer=self.cfg.tokenizer,
                override_data_prep=self.cfg.override_data_prep,
                data_split_percent=self.cfg.data_split_percent,
            )
        )
        return dm.test_dataloader()

    def test(self):
        test_csv_out_data = []
        headers = ["target", "prediction"]
        text_csv_out_path = f"predictions_{self.cfg.dataset_config.name}_{self.cfg.dataset_config.task_name}_{self.cfg.data_split_percent}.csv"
        all_targets = []
        all_predictions = []
        for batch in tqdm(self.dataloader):
            gen = self.cfg.model.generate(
                inputs=batch.context_tokens.cuda(),
                attention_mask=batch.context_attention_masks.cuda(),
                do_sample=True,
                top_k=50,
                top_p=0.94,
                max_length=self.cfg.generate_max_len,
                temperature=0.5,
                eos_token_id=self._get_token_id(SpecialTokens.end_target),
                pad_token_id=self._get_token_id(TokenizerTokens.pad_token),
                bos_token_id=self._get_token_id(SpecialTokens.begin_target),
            )
            gen_without_context = gen[:, self.cfg.max_token_len :]
            pred_text = self.cfg.tokenizer.batch_decode(
                gen_without_context, skip_special_tokens=False
            )
            pred_text_no_pad = [self._remove_padding(text) for text in pred_text]
            self.tod_metrics.add_batch(
                references=batch.targets, predictions=pred_text_no_pad
            )
            all_targets.append(batch.targets)
            all_predictions.append(pred_text_no_pad)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        test_csv_out_data = np.column_stack([all_targets, all_predictions])
        utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
        self.cfg.logger.info(str(self.tod_metrics))
        self.tod_metrics.visualize(self.cfg.predictions_log_dir)


@hydra.main(config_path="../config/inference/", config_name="inference")
def hydra_start(cfg: DictConfig) -> None:
    inf = Inference(InferenceConfig(**cfg))
    inf.run()


if __name__ == "__main__":
    hydra_start()
