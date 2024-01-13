######################################################
## ALBERT Base Model                                ##
## 1. Pre-trained ALBERT from Hugging Face          ##
## 2. Fine Tuning on our Primary Dataset            ##
########## Prepared by: Wilson Widhyadana ############
######################################################

import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AlbertForSequenceClassification, PreTrainedTokenizerBase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_FILE = "../Datasets/clean_data/X_train_dev.csv"
TRAIN_LABEL_FILE = "../Datasets/clean_data/y_train_dev.csv"
TEST_DATA_FILE = "../Datasets/clean_data/X_test.csv"
TEST_LABEL_FILE = "../Datasets/clean_data/y_test.csv"
RANDOM_SEED = 4248
MAX_TOKEN_LENGTH = 150
BATCH_SIZE = 128
NUM_EPOCHS = 10

print(f"Device: {DEVICE}")

class SarcasmDataset(Dataset):
    def __init__(
            self, 
            data_path: str | os.PathLike = None,
            label_path: str | os.PathLike = None,
            data: pd.Series = None,
            label: pd.Series = None,
            tokenizer: AutoTokenizer | PreTrainedTokenizerBase = None,
            max_length: int = MAX_TOKEN_LENGTH
        ):
        """
        Args:
            data_path (str | os.PathLike): Path to a .csv file containing the data features. Defaults to None.
            label_path (str | os.PathLike): Path to a .csv file containing the data labels. Defaults to None.
            data (pd.Series): Actual data. Defaults to None.
            label (pd.Series): Actual labels. Defaults to None.
            tokenizer (AutoTokenizer | PreTrainedTokenizerBase): Tokenizer to be used. Defaults to None.
            max_length (int): Maximum token sequence length for a single sequence. Defaults to MAX_TOKEN_LENGTH.
        """

        # Do some checking of arguments
        if not (data_path or data is not None) or not (label_path or label is not None):
            raise ValueError("Invalid data and label arguments.")

        if data_path and data is not None:
            raise ValueError("Only one of data_path or data must be set.")

        if label_path and label is not None:
            raise ValueError("Only one of label_path or label must be set.")

        if not tokenizer:
            raise ValueError("tokenizer must not be None.")

        if data_path and label_path:
            self.data_path = data_path
            self.label_path = label_path
            self.data = pd.read_csv(data_path)["text"]
            self.data = tokenizer(
                self.data, 
                truncation=True, 
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.input_ids = self.data["input_ids"]
            self.token_type_ids = self.data["token_type_ids"]
            self.attention_mask = self.data["attention_mask"]
            self.label = torch.tensor(pd.read_csv(label_path)["label"].tolist())
            self.label = nn.functional.one_hot(self.label, num_classes=2)
            self.label = self.label.type(torch.float32)
        elif (data_path and not label_path) or (not data_path and label_path):
            raise ValueError("Both data_path and label_path must be set.")

        if data is not None and label is not None:
            self.data = tokenizer(
                data, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt")
            self.input_ids = self.data["input_ids"]
            self.token_type_ids = self.data["token_type_ids"]
            self.attention_mask = self.data["attention_mask"]
            self.label = torch.tensor(label)
            self.label = nn.functional.one_hot(self.label, num_classes=2)
            self.label = self.label.type(torch.float32)
        else:
            raise ValueError("Both data and label must be set.")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        return (
            {
                "input_ids": self.input_ids[idx], 
                "token_type_ids": self.token_type_ids[idx], 
                "attention_mask": self.attention_mask[idx]    
            }, 
            self.label[idx]
        )

train_dev_data_series = pd.read_csv(TRAIN_DATA_FILE)["text"].tolist()
train_dev_label_series = pd.read_csv(TRAIN_LABEL_FILE)["label"].tolist()
test_data_series = pd.read_csv(TEST_DATA_FILE)["text"].tolist()
test_label_series = pd.read_csv(TEST_LABEL_FILE)["label"].tolist()

X_train, X_dev, y_train, y_dev = train_test_split(
    train_dev_data_series,
    train_dev_label_series,
    test_size=0.11,
    random_state=RANDOM_SEED
)


tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-CoLA")
train_dataset = SarcasmDataset(
    data=X_train, 
    label=y_train, 
    tokenizer=tokenizer, 
    max_length=MAX_TOKEN_LENGTH
)
dev_dataset =  SarcasmDataset(
    data=X_dev, 
    label=y_dev, 
    tokenizer=tokenizer, 
    max_length=MAX_TOKEN_LENGTH
)
test_dataset = SarcasmDataset(
    data=test_data_series,
    label=test_label_series,
    tokenizer=tokenizer,
    max_length=MAX_TOKEN_LENGTH
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 0,
    "train_losses": [],
    "train_f1": [],
    "train_acc": [],
    "dev_losses": [],
    "dev_f1": [],
    "dev_acc": []
}

for epoch in range(NUM_EPOCHS):
    running_loss, running_f1, running_acc = 0, 0, 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch[0]["input_ids"].to(DEVICE)
        token_type_ids = batch[0]["token_type_ids"].to(DEVICE)
        attention_mask = batch[0]["attention_mask"].to(DEVICE)
        labels = batch[1].to(DEVICE)

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(outputs, dim=1)
        target_labels = torch.argmax(labels, dim=1)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        cpu_labels = target_labels.cpu().detach().numpy()
        cpu_preds = predictions.cpu().detach().numpy()
        cur_f1 = f1_score(cpu_labels, cpu_preds, average="micro")
        cur_acc = accuracy_score(cpu_labels, cpu_preds)

        running_loss += loss.item()
        running_f1 += cur_f1
        running_acc += cur_acc

        if step % 10 == 9:
            avg_loss = running_loss / 10
            avg_f1 = running_f1 / 10
            avg_acc = running_acc / 10

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"Train Loss: {avg_loss:.4f} | Train F1: {avg_f1:.4f} | Train Acc: {avg_acc:.4f}")

            checkpoint["train_losses"].append(avg_loss)
            checkpoint["train_f1"].append(avg_f1)
            checkpoint["train_acc"].append(avg_acc)

            running_loss, running_f1, running_acc = 0, 0, 0

        loss.backward()
        del loss
        optimizer.step()

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"] = epoch + 1

    model.eval()
    dev_run_loss, dev_run_f1, dev_run_acc = 0, 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            input_ids = batch[0]["input_ids"].to(DEVICE)
            token_type_ids = batch[0]["token_type_ids"].to(DEVICE)
            attention_mask = batch[0]["attention_mask"].to(DEVICE)
            labels = batch[1].to(DEVICE)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
            predictions = torch.argmax(outputs, dim=1)
            target_labels = torch.argmax(labels, dim=1)

            loss = loss_fn(outputs, labels)
            cpu_labels = target_labels.cpu().detach().numpy()
            cpu_preds = predictions.cpu().detach().numpy()
            cur_f1 = f1_score(cpu_labels, cpu_preds, average="micro")
            cur_acc = accuracy_score(cpu_labels, cpu_preds)

            dev_run_loss += loss.item()
            del loss
            dev_run_f1 += cur_f1
            dev_run_acc += cur_acc

            if step % 10 == 9:
                avg_loss = dev_run_loss / 10
                avg_f1 = dev_run_f1 / 10
                avg_acc = dev_run_acc / 10

                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
                print(f"Dev Loss: {avg_loss:.4f} | Dev F1: {avg_f1:.4f} | Dev Acc: {avg_acc:.4f}")

                checkpoint["dev_losses"].append(avg_loss)
                checkpoint["dev_f1"].append(avg_f1)
                checkpoint["dev_acc"].append(avg_acc)

                dev_run_loss, dev_run_f1, dev_run_acc = 0, 0, 0
    
    torch.save(checkpoint, f"albert_baseline_model_{epoch+1}.pt")
