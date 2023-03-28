import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from datasets import load_dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

class MovieRecommendationModel(LightningModule):
    def __init__(self, model_name_or_path, peft_config, train_data, test_data, batch_size, max_length, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model = get_peft_model(self.model, peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.dataset = load_dataset(
            'csv', data_files={
                "train": train_data,
                "validation": test_data,
            }, 
            cache_dir="./cache")

        self.processed_datasets = self.preprocess_datasets()
        self.train_dataset = self.processed_datasets["train"]
        self.eval_dataset = self.processed_datasets["validation"]

    def preprocess_datasets(self):
        def preprocess_function(examples):
            inputs = [doc for doc in examples["question"]]
            model_inputs = self.tokenizer(
                inputs, max_length=self.hparams.max_length, padding=True, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["answer"], max_length=self.hparams.max_length, padding=True, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = self.dataset.map(
            preprocess_function,
            batched=True,
            num_proc=16,
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataset) * self.hparams.num_epochs),
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, collate_fn=self.data_collator, batch_size=self.hparams.batch_size, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset, collate_fn=self.data_collator, batch_size=self.hparams.batch_size, pin_memory=True
        )


def main():
    model_name_or_path = "google/flan-t5-xxl"
    batch_size = 2
    max_length = 512
    lr = 1e-4
    num_epochs
