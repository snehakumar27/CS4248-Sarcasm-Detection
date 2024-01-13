######################################################
## DistilBERT Base Model                            ##
## 1. Pre-trained DistilBERT from Hugging Face      ##
## 2. Fine Tuning on our Primary Dataset            ##
## 3. Albert                                        ##
############# Prepared by: Sneha Kumar ###############
######################################################

""" # Import Data & Libraries#"""
# import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# import data and split into training and validation
X_train_valid_directory = "../Datasets/clean_data/X_train_dev.csv"
y_train_valid_directory = "../Datasets/clean_data/y_train_dev.csv"
X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)

""" # Model Training# """
# set device
if torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("GPU used")
else: 
    device = torch.device("cpu")
    print("GPU not found")

## tokenization
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
max_len = 150

x_train = list(X_train)
x_valid = list(X_valid)

print('Encodings Started')
train_encodings = tokenizer(x_train, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
valid_encodings = tokenizer(x_valid, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
print("Encodings Created")

## create tensor dataset and data loader
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(list(y_train), dtype=torch.float32))
valid_dataset = TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'], torch.tensor(list(y_valid), dtype=torch.float32))

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print("Data Loaded")

## Training parameters
print("Training Prep")
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=1)
model.to(device)

loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

epochs = 3

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 0,
    'step':0,
    'train_losses': [],
    'train_f1_scores': [],
    'valid_losses': [],
    'valid_f1_scores': []
}


## Model Training ##
print("Training started")
for epoch in range(epochs):
    running_loss = 0
    running_f1 = 0
    model.train()
    for step, batch in enumerate(train_loader, 0):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask).logits.view(-1)

        loss = loss_func(outputs, labels)
        running_loss += loss.item()

        f1_batch = f1_score(labels.cpu().numpy(), np.round(torch.sigmoid(outputs.detach()).cpu().numpy()), average = 'binary')
        running_f1 += f1_batch

        if step % 10 == 9:
            avg_loss = running_loss / 10
            avg_f1 = running_f1 / 10

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {avg_loss:.4f} | Train F1: {avg_f1:.4f}")

            checkpoint['train_losses'].append(avg_loss)
            checkpoint['train_f1_scores'].append(avg_f1)

            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['step'] = step

            torch.save(checkpoint, f'dbert_base_model_{epoch + 1}.pt')

            running_loss = 0.0
            running_f1 = 0

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['epoch'] = epoch + 1

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_f1 = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask).logits.view(-1)

            loss = loss_func(outputs, labels)
            val_loss += loss.item()

            f1_batch = f1_score(labels.cpu().numpy(), np.round(torch.sigmoid(outputs).cpu().numpy()), average = 'binary')
            val_f1 += f1_batch

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_f1 = val_f1 / len(valid_loader)

        print(f"Validation Loss: {avg_val_loss:.4f} | Validation F1: {avg_val_f1:.4f}")
        checkpoint['valid_losses'].append(avg_val_loss)
        checkpoint['valid_f1_scores'].append(avg_val_f1)

    torch.save(checkpoint, f'dbert_base_model_{epoch+1}_complete.pt')
