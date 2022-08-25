import os
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from hydra_configs import DataModuleConfig, TrainerConfig
from my_datamodules import MyCollators, MyDataModule, Steps


class MyTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def run(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModel.from_pretrained(self.config.model_name)
        model = model.cuda()

        dm = MyDataModule(DataModuleConfig.from_trainer_config(self.config))
        dm.setup()
        self.train(model, dm)

    def train(self, model: AutoModel, dm: MyDataModule):
        training_args = TrainingArguments(
            output_dir=self.config.out_root,
            num_train_epochs=self.config.epochs,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.config.logging_dir,
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dm.datasets[Steps.train],
            eval_dataset=dm.datasets[Steps.val],
            data_collator=MyCollators.turn_collator,
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.out_root)
        print("out dir ", os.getcwd())


@hydra.main(config_path="../config/trainer", config_name="trainer")
def hydra_start(cfg: DictConfig) -> None:
    msc = MyTrainer(TrainerConfig(**cfg))
    msc.run()


if __name__ == "__main__":
    hydra_start()
