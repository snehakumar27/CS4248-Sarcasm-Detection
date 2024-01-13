#### RoBERTa ####
## Prepared by: Matthias Koh Yong An

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler

from transformers import AutoTokenizer, RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification

X_train_valid_directory = "../Datasets/clean_data/X_train_dev.csv"
y_train_valid_directory = "../Datasets/clean_data/y_train_dev.csv"
X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
#X_train_dev = X_train_dev[:50000] #subset taken when required due to computation limit and for faster results/fine tuning
#y_train_dev = y_train_dev[:50000] 
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)

# Tokenizer and dataset formulation
model_name = 'roberta-large' #roberta-base'
tokenizer =  RobertaTokenizer.from_pretrained(model_name)

max_len = 60
batch_size = 48

train_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length. 
valid_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = t.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

device = t.device("cuda") if t.cuda.is_available() else "cpu"

# MODEL setup
model = RobertaForSequenceClassification.from_pretrained(model_name)
model.to(device)

# Define the optimizer and loss function
lr = 1e-5 #0.001
optimizer = t.optim.AdamW(model.parameters(), 
                          lr=lr, 
                          weight_decay= 0.01, #0.004
                          betas = ([0.9,0.999]), 
                          eps=1e-07)
                          
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(4):
  print(epoch)
  model.train()
  running_loss = 0.0

  for i, (data, labels) in enumerate(train_loader, 0):
    print(i)
    input_ids = data['input_ids'].squeeze(dim=1).to(device) # need squeeze dim is wrong now
    attention_mask = data['attention_mask'].to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) #attention_mask=attention_mask
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    running_loss +=loss.item()

    # # print loss value every 120 iterations and reset running loss
    if i % 120 == 119: 
         print('[%d, %5d] loss: %.3f' %
         (epoch + 1, i + 1, running_loss / 120)) 
         running_loss = 0.0
    
  # validation
  model.eval()
  with t.no_grad():
    val_running_loss = 0.0
    predictions = []
    true_labels = []
    for j, (val_data, val_labels) in enumerate(valid_loader, 0):
      val_input_ids = val_data['input_ids'].squeeze(dim=1).to(device)
      val_attention_mask = val_data['attention_mask'].to(device)
      val_labels = val_labels.to(device)

      val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
      val_loss = val_outputs.loss
      val_running_loss += val_loss.item()

      val_predictions = t.argmax(val_outputs.logits, dim=1)
      predictions.extend(val_predictions.cpu().detach().numpy())
      true_labels.extend(val_labels.cpu().detach().numpy())

    val_f1 = f1_score(true_labels, predictions)
    print(f'Validation F1 Score: {val_f1}')
