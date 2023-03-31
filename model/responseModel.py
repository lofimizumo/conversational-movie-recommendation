import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from datasets import load_dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from peft import LoraConfig, TaskType, get_peft_model
from model.responseConfig import MovieResponseConfig


class MovieResponseModel(LightningModule):
    '''
    Movie Recommendation Model
    '''

    def __init__(self, model_name_or_path, data_path, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)

        self.dataset = load_dataset('csv', data_files=data_path, delimiter='@', column_names=[
            "movie_name", "question", "answer"], cache_dir="./cache", split='train')

        self.dataset = self.dataset.train_test_split(
            test_size=0.2, shuffle=True)

        self.processed_datasets = self.preprocess_datasets()
        self.train_dataset = self.processed_datasets["train"]
        self.eval_dataset = self.processed_datasets["test"]

        self.config = config
    

    def setPeftModel(self, peft_config):
        self.model = get_peft_model(self.model, peft_config)

    def preprocess_datasets(self):
        def preprocess_function(examples):
            inputs = [doc for doc in examples["question"]]
            model_inputs = self.tokenizer(
                inputs, max_length=2048, padding=True, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["answer"], max_length=2048, padding=True, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = self.dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        return processed_datasets

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataset) *
                                self.config.num_epochs),
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, collate_fn=self.data_collator, batch_size=self.config.batch_size, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset, collate_fn=self.data_collator, batch_size=self.config.batch_size, pin_memory=True
        )


def main():
    model_name = "google/flan-t5-small"
    data = "./data/prompt_answer_150.csv"
    config = MovieResponseConfig(
        batch_size=15,
        max_length=2048,
        lr=1e-4,
        num_epochs=15
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    model = MovieResponseModel(
        model_name_or_path=model_name,
        data_path=data,
        config=config,
    )

    # model.setPeftModel(peft_config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-checkpoint",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
# write a script to load the model "best-checkpoint.ckpt" and do some inference to generate some movie recommendations.
