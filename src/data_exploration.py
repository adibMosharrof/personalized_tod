import humps
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotmap import DotMap
from omegaconf import DictConfig
from tqdm import tqdm

from babi_dataclasses import BabiTask1SystemColumns
from hydra_configs import DataModuleConfig, DataExplorationConfig
from my_datamodules import MyDataModule, Steps


class DataExploration:
    col_names = DotMap(step="step", count="count")

    def __init__(self, config: DataExplorationConfig):
        self.project_root = config.project_root
        self.data_root = config.data_root
        self.out_root = config.out_root
        self.dataset_name = config.dataset_name
        self.task_name = config.task_name

    def get_system_df(self, df: pd.DataFrame, data_index: list[int], step: str):
        indexes = [i - 1 for i in data_index]
        df_api = df.iloc[indexes]
        system_df = df_api.system.str.split(" ", expand=True)
        system_df.columns = BabiTask1SystemColumns.values()
        system_df = system_df.assign(step=step)
        return system_df[system_df.columns[1:]]

    def plot_grouped_bar_chart(
        self, data: pd.DataFrame, col: str, dataset_name: str, task_name: str
    ):
        plt.style.use("ggplot")
        sns.set(style="whitegrid")

        plt.figure()
        g = sns.catplot(
            data=data,
            kind="bar",
            hue=self.col_names.step,
            x=col,
            y=self.col_names.count,
        )
        g.set_axis_labels(
            x_label=humps.pascalize(col), y_label=humps.pascalize(self.col_names.count)
        )
        g.despine(left=True)
        g._legend.remove()
        plt.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
        plt.title(
            f"{humps.pascalize(dataset_name)} {humps.pascalize(task_name)} {humps.pascalize(col)} Distribution"
        )
        plt.tight_layout()
        plt.savefig(f"{self.out_root}/{dataset_name}_{task_name}{col}_distribution.png")
        plt.close()

    def plot_system_distribution(
        self, df: pd.DataFrame, dataset_name: str, task_name: str
    ):
        cols = df.columns[:-1]
        for col in cols:
            data = (
                df.groupby([col, self.col_names.step])
                .size()
                .to_frame(self.col_names.count)
                .reset_index()
            )
            self.plot_grouped_bar_chart(data, col, dataset_name, task_name)

    def run(self):
        dm = MyDataModule(
            DataModuleConfig(dataset_name=self.dataset_name, task_name=self.task_name)
        )
        datasets = dm.setup()
        system_df = []
        for (step_key, step_val), (_, vals) in zip(Steps.items(), datasets.items()):
            dataset = vals.data
            dialog_index = vals.dialog_index
            df = pd.DataFrame(dataset)
            user_avg = df.user.apply(len).mean()
            system_avg = df.system.apply(len).mean()
            df_dialogs = df.groupby("dialog_id")
            turn_len_avg = df_dialogs.apply(len).mean()
            print(
                f"Step: {step_key}, user avg: {user_avg}, system avg: {system_avg}, turn len avg: {turn_len_avg}"
            )
            system_df.append(self.get_system_df(df, dialog_index, step_key))
        all_system_df = pd.concat(system_df, axis=0)
        self.plot_system_distribution(all_system_df, self.dataset_name, self.task_name)


@hydra.main(config_path="../config/", config_name="data_exploration")
def hydra_start(cfg: DictConfig) -> None:
    msc = DataExploration(DataExplorationConfig(**cfg))
    msc.run()


if __name__ == "__main__":
    hydra_start()
