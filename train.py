from transformers import (GPT2TokenizerFast, AdamW, get_scheduler)
import torch
from model import GPTNeoPromptTuningLM
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from datasets import Dataset as HDataset

class NewsDataset(Dataset):
    def __init__(self, file_path: str="readerbench/AlephNews", split: str="train"):
        self.file_path = file_path
        self.instances = []

        dataset = load_dataset(file_path)
        df_pandas = pd.DataFrame(dataset[split])
        df_pandas.dropna(inplace=True)
        dataset = HDataset.from_pandas(df_pandas)
        for line in dataset:
            paragraph = line["paragraphs"][0]
            summary = line["summary"][0]
            instance = {
                        "sentence": paragraph+"\nSUMMARY: "+summary+"<|endoftext|>",
                       }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class GPTSummarization(pl.LightningModule):
    def __init__(self, model_name:str="dumitrescustefan/gpt-neo-romanian-780m",
                 num_train_epochs:int=3,
                 weight_decay:float=0.01,
                 learning_rate:float=0.01,
                 num_warmup_steps:int=0,
                 n_prompt_tokens:int=20,
                 init_from_vocab:bool=True):
        super().__init__()
        self.num_train_epochs=num_train_epochs
        self.weight_decay=weight_decay
        self.learning_rate=learning_rate
        self.num_warmup_steps=num_warmup_steps
        self.max_train_steps=num_train_epochs
        self.n_prompt_tokens=n_prompt_tokens
        self.init_from_vocab=init_from_vocab
        self.model_name = model_name

        self.model = GPTNeoPromptTuningLM.from_pretrained(self.model_name,
                                                          n_tokens=self.n_prompt_tokens,
                                                          initialize_from_vocab=self.init_from_vocab)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        # because reshaping, need to refreeze embeddings
        for name, param in self.model.named_parameters():
            if name == "transformer.wte.weight":
                param.requires_grad=False

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def my_collate(self, batch):
        sentences = []
        for instance in batch:
            sentences.append(instance["sentence"])

        sentences_batch = self.tokenizer(sentences, padding="max_length", max_length=472, truncation=True, return_tensors="pt")
        return sentences_batch

    def forward(self, sentence):
        outputs = self.model(input_ids=sentence["input_ids"], attention_mask=sentence["attention_mask"],
                             labels=sentence["input_ids"], )
        loss, logits = outputs.loss, outputs.logits
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.train_loss.append(loss.detach().cpu().numpy())
        if loss == 0:
            return None
        return {"train_loss":loss}

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.val_loss.append(loss.detach().cpu().numpy())
        if loss == 0:
            return None
        return {"valid_loss":loss}

    def test_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.test_loss.append(loss.detach().cpu().numpy())
        if loss == 0:
            return None
        return {"test_loss":loss}

    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n == "soft_prompt.weight"],
                "weight_decay": self.weight_decay,
             }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # data
    train_dataset = NewsDataset(split="train")
    test_dataset = NewsDataset(split="test")
    valid_dataset = NewsDataset(split="validation")

    model = GPTSummarization()
    trainer = pl.Trainer()

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True, collate_fn=model.my_collate, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True )
    val_loader = DataLoader(valid_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True)

    trainer.fit(model, train_loader, val_loader)
    save_dir_path = "./soft_prompt"
    model.model.save_soft_prompt(save_dir_path)
    #trainer.test(test_dataloaders=test_loader)

if __name__ == "__main__":
    cli_main()
