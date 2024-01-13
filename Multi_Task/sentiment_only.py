######################################################
## Multi-task  (get benchmark for sentiment only)   ##
############# Prepared by: Sneha Kumar ###############
######################################################
# import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel

## read data
# sarcasm data

# sentiment data
sent_train_directory = "../Datasets/clean_data/sent_train.csv"
sent_valid_directory = "../Datasets/clean_data/sent_test.csv"
sent_train = pd.read_csv(sent_train_directory)
sent_valid = pd.read_csv(sent_valid_directory)

num_rows = 25000
sent_train = sent_train.sample(n = num_rows, random_state = 4248)
sent_x_train = list(sent_train['text'])
sent_y_train = list(sent_train['label'])
sent_x_valid = list(sent_valid['text'])
sent_y_valid = list(sent_valid['label'])

## data preprocessing 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU used")
else:
    device = torch.device("cpu")
    print("GPU not found")

# sentiment encodings
model_name = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
max_len = 150

print('Encodings Started')
sent_train_encodings = tokenizer(sent_x_train, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
sent_valid_encodings = tokenizer(sent_x_valid, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
print("Encodings Created")

sent_train_dataset = TensorDataset(sent_train_encodings['input_ids'], sent_train_encodings['attention_mask'], torch.tensor(sent_y_train, dtype=torch.float32))
sent_valid_dataset = TensorDataset(sent_valid_encodings['input_ids'], sent_valid_encodings['attention_mask'], torch.tensor(sent_y_valid, dtype=torch.float32))

batch_size = 128
sent_train_loader = DataLoader(sent_train_dataset, batch_size=batch_size, shuffle=True)
sent_valid_loader = DataLoader(sent_valid_dataset, batch_size=batch_size, shuffle=False)
print("Data Loaded")

## create model 
class MultiTask_Distil(torch.nn.Module):
    def __init__(self, hidden_dim = 768, drp_rate = 0.1):
        super(MultiTask_Distil, self).__init__()
        self.base = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, hidden_dim)
        self.dropout = torch.nn.Dropout(drp_rate)
        self.classifier_1 = torch.nn.Linear(hidden_dim, 1)
        self.classifier_2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, task_id):
        base_output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = base_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        if task_id == 1:
          output = self.classifier_1(pooler)
        elif task_id == 2:
          output = self.classifier_2(pooler)
        else:
            assert False, 'Bad Task ID passed'
        return output
    
model = MultiTask_Distil(drp_rate = 0.4)
model.to(device)

loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

epochs = 3

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_sent_losses': [],
    'train_sent_f1_scores': [],
    'valid_sent_losses': [], 
    'valid_sent_f1_scores': []
}

print("Training started")

## train model
for epoch in range(epochs):
    running_loss = 0
    running_f1_sentiment = 0
    model.train()
    for step, sent_batch in enumerate(sent_train_loader, 0):
        sent_input_ids = sent_batch[0].to(device)
        sent_attention_mask = sent_batch[1].to(device)
        sent_labels = sent_batch[2].to(device)
        sent_outputs = model(sent_input_ids, sent_attention_mask, task_id = 2).view(-1)

        sent_loss = loss_func(sent_outputs, sent_labels)
        sent_f1 = f1_score(sent_labels.cpu().numpy(), np.round(torch.sigmoid(sent_outputs).cpu().detach().numpy()), average = 'binary')

        running_loss += sent_loss.item()
        running_f1_sentiment += sent_f1

        optimizer.zero_grad()
        sent_loss.backward()
        optimizer.step()

        if step % 49 == 48:
            avg_loss = running_loss / 49
            avg_sent_f1 = running_f1_sentiment / 49

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss Weighted Sum: {avg_loss:.4f} | Train Sentiment F1: {avg_sent_f1:.4f}")

            checkpoint['train_sum_losses'].append(avg_loss)
            checkpoint['train_sent_f1_scores'].append(avg_sent_f1)
            torch.save(checkpoint, f'multi-task_{epoch + 1}.pt')

            running_loss = 0.0
            running_f1_sentiment = 0

            model.eval()
            with torch.no_grad():
                sent_val_loss = 0.0
                sent_val_f1 = 0.0
                for batch in sent_valid_loader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[2].to(device)

                    outputs = model(input_ids, attention_mask, task_id = 2).view(-1)

                    loss = loss_func(outputs, labels)
                    sent_val_loss += loss.item()

                    f1_batch = f1_score(labels.cpu().numpy(), np.round(torch.sigmoid(outputs).cpu().numpy()), average = 'binary')
                    sent_val_f1 += f1_batch

                avg_sent_val_loss = sent_val_loss / len(sent_valid_loader)
                avg_sent_val_f1 = sent_val_f1 / len(sent_valid_loader)

                print(f"Sentiment Validation Loss: {avg_sent_val_loss:.4f} | Sentiment Validation F1: {avg_sent_val_f1:.4f}")
                checkpoint['valid_sent_losses'].append(avg_sent_val_loss)
                checkpoint['valid_sent_f1_scores'].append(avg_sent_val_f1)
            torch.save(checkpoint, f'multi-task_{epoch + 1}.pt')
            model.train()
            
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, f'multi-task_complete_{epoch + 1}.pt')