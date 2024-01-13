#### Validation and Test Evaluation ####
## Prepared by: Sneha Kumar
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

model_path = "dbert_base_model_3_complete.pt"       # path to the model being evaluated
X_train_valid_directory = "../../X_train_dev.csv"
y_train_valid_directory = "../../y_train_dev.csv"
X_test_directory = "../../X_test.csv"
y_test_directory = "../../y_test.csv"

X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)

X_test = pd.read_csv(X_test_directory)
y_test = pd.read_csv(y_test_directory)

## Prepare Validation & Test Data 
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
max_len = 150

x_valid = list(X_valid)
x_test = list(X_test['text'])

valid_encodings = tokenizer(x_valid, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
test_encodings = tokenizer(x_test, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')

valid_dataset = TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'], torch.tensor(list(y_valid), dtype=torch.float32))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(list(y_test['label']), dtype=torch.float32))

batch_size = 128
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Init Model Class and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=1) # replace with the model being evaluated for (create and add the class if it's a custom model)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

## Model Evaluation
model.eval()
valid_preds, valid_labels = [], []
test_preds, test_labels = [], []

with torch.no_grad():
    val_f1 = 0.0
    for batch in valid_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask).logits.view(-1)

        valid_preds.extend(np.round(torch.sigmoid(outputs).cpu().numpy()))
        valid_labels.extend(labels.cpu().numpy())

        f1_batch = f1_score(labels.cpu().numpy(), np.round(torch.sigmoid(outputs).cpu().numpy()), average = 'binary')
        val_f1 += f1_batch
        
    avg_val_f1 = val_f1 / len(valid_loader)
    print("Validation F1: {:.4f}".format(avg_val_f1))

    val_f1 = 0.0
    for batch in test_loader: 
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask).logits.view(-1)

        test_preds.extend(np.round(torch.sigmoid(outputs).cpu().numpy()))
        test_labels.extend(labels.cpu().numpy())

        f1_batch = f1_score(labels.cpu().numpy(), np.round(torch.sigmoid(outputs).cpu().numpy()), average = 'binary')
        val_f1 += f1_batch
        
    avg_val_f1 = val_f1 / len(valid_loader)
    print("Test F1: {:.4f}".format(avg_val_f1))


## Confusion Matrix Plotting
valid_cm = confusion_matrix(valid_preds, valid_labels)
test_cm = confusion_matrix(test_preds, test_labels)

fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# Plot confusion matrix for validation set
sns.heatmap(valid_cm, annot=True, fmt='d', ax=ax[0])
ax[0].set_title('Validation Set')
ax[0].set_xlabel('True')
ax[0].set_ylabel('Predicted')

# Plot confusion matrix for test set
sns.heatmap(test_cm, annot=True, fmt='d', ax=ax[1])
ax[1].set_title('Test Set')
ax[1].set_xlabel('True')
ax[1].set_ylabel('Predicted')

# Save the figure
plt.savefig('Evaluations_and_Plots/conf_mat.png')

