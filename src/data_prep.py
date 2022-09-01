import utils
import hydra
import numpy as np
from omegaconf import DictConfig

from hydra_configs import DataPrepConfig
from tod_utils import Steps
import pandas as pd

from task_dataclasses import BaseTask
from tod_dataclasses import TodContext, TodTarget, TodTurn, get_csv_data_path


class DataPrep:
    def __init__(self, cfg: DataPrepConfig):
        self.cfg = cfg

    def _prepare_turn(self, turn: BaseTask, turns: list[BaseTask], index: int):
        if index == 0:
            prev_turns = []
        else:
            prev_turns = turns[:index]
        context = TodContext(turns=prev_turns, current_user_utterance=turn.user)
        target = TodTarget(turn)

        return TodTurn(
            context=context,
            target=target,
            dialog_id=turn.dialog_id,
            turn_id=turn.turn_id,
        )

    def _prepare_dialog(self, turns: list[BaseTask]) -> list[TodTurn]:
        tod_turns = []
        for i, turn in enumerate(turns):
            tod_turn = self._prepare_turn(turn, turns, i)
            tod_turns.append(tod_turn.to_csv_row())
        return tod_turns

    def run(self):
        for (step_key, step_val), split_percent in zip(
            Steps.items(), self.cfg.data_split_percent
        ):
            step_dir = self.cfg.processed_data_root / step_key
            step_dir.mkdir(exist_ok=True, parents=True)
            out_path = get_csv_data_path(
                self.cfg.dataset_name,
                self.cfg.task_name,
                step_key,
                split_percent,
                self.cfg.processed_data_root,
            )
            if out_path.exists() and not self.cfg.override_data_prep:
                print(
                    f"CSV file for {step_key} already exists, so skipping data preparation"
                )
                continue
            turn_data = []
            raw_data = self.cfg.dataset_config.task_class.read_raw_data(
                self.cfg.dataset_config.get_task_file_path(step_val), split_percent
            )
            df = pd.DataFrame(raw_data.data)
            df_dialog_groups = pd.DataFrame(df).groupby("dialog_id")
            for _, dialog_group in df_dialog_groups:
                tasks = [
                    self.cfg.dataset_config.task_class(*item)
                    for item in dialog_group.values.tolist()
                ]
                tod_turns = self._prepare_dialog(tasks)
                turn_data.append(tod_turns)

            utils.write_csv(
                ["dialog_id", "turn_id", "context", "target"],
                np.concatenate(turn_data, axis=0),
                out_path,
            )


@hydra.main(config_path="../config/", config_name="data_prep")
def hydra_start(cfg: DictConfig) -> None:
    stdp = DataPrep(DataPrepConfig(**cfg))
    stdp.run()


if __name__ == "__main__":
    hydra_start()
