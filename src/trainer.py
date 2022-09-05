from lib2to3.pgen2.tokenize import tokenize
import os
import hydra
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from hydra_configs import DataModuleConfig, InferenceConfig, TrainerConfig
from inference import Inference
from my_datamodules import MyDataModule, Steps


class MyTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg

    def run(self):

        dm = MyDataModule(
            DataModuleConfig.from_trainer_config(self.cfg, tokenizer=self.cfg.tokenizer)
        )
        self.train_model(dm)

    def train_model(self, dm: MyDataModule):
        training_args = TrainingArguments(
            output_dir=str(self.cfg.pretrain_out_root),
            num_train_epochs=self.cfg.pretrain_epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            dataloader_pin_memory=True,
        )
        if self.cfg.pretrain_model:
            model = self.cfg.pretrain_model
            self.cfg.logger.info("loaded pretrained model")
        else:
            model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
            model.resize_token_embeddings(len(self.cfg.tokenizer))
            model = model.cuda()

            pre_trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dm.datasets[Steps.train],
                eval_dataset=dm.datasets[Steps.val],
                data_collator=dm.pretrain_collator,
            )
            pre_trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
            pre_trainer.train()
            pre_trainer.save_model()
            print("pretraining complete")
            print("-" * 50)

        # model_train = GPT2LMHeadModel.from_pretrained(self.cfg.pretrain_out_root)
        training_args.output_dir = str(self.cfg.train_out_root)
        training_args.num_train_epochs = self.cfg.train_epochs
        trainer = Trainer(
            # model=model_train,
            model=model,
            args=training_args,
            train_dataset=dm.datasets[Steps.train],
            eval_dataset=dm.datasets[Steps.val],
            data_collator=dm.training_collator,
        )
        trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
        trainer.train()
        trainer.save_model()

        print("training complete")
        print("-" * 50)
        self.cfg.tokenizer.save_pretrained(self.cfg.out_root)
        self.cfg.logger.info(f"out dir {os.getcwd()}")

        if self.cfg.should_test:
            inf = Inference(
                InferenceConfig.from_trainer_config(self.cfg, model, self.cfg.tokenizer)
            )
            inf.run()


@hydra.main(config_path="../config/trainer", config_name="trainer")
def hydra_start(cfg: DictConfig) -> None:
    msc = MyTrainer(TrainerConfig(**cfg))
    msc.run()


if __name__ == "__main__":
    hydra_start()
