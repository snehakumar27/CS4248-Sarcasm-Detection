#### RobXLMRob ####
## Prepared by: Matthias Koh Yong An

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler

from transformers import AutoTokenizer, AlbertTokenizer,AlbertTokenizerFast, AlbertForSequenceClassification, RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW, RobertaModel, XLMRobertaModel,AlbertModel

X_train_valid_directory = "../Datasets/clean_data/X_train_dev.csv"
y_train_valid_directory = "../Datasets/clean_data/y_train_dev.csv"
X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
#X_train_dev = X_train_dev[:50000] #subset taken when required due to computation limit and for faster results/fine tuning
#y_train_dev = y_train_dev[:50000]
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)

# Tokenizer and dataset formulation
model_name = 'roberta-large' #'xlm-roberta-large' #'roberta-large' #roberta-base'
tokenizer =  AutoTokenizer.from_pretrained(model_name)

max_len = 60
batch_size = 48

train_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = t.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

device = t.device("cuda") if t.cuda.is_available() else "cpu"

# load the pre-trained models
roberta_model = RobertaModel.from_pretrained("roberta-large")
xlmroberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

# define pooler output
pooler = nn.AdaptiveAvgPool1d(1) # output of size 1

# define layer normalization
layer_norm = nn.LayerNorm(normalized_shape=1024, eps=1e-6) #768 for small #1024 for eg xlmrobert-large checkpoint

# define multi-sample dropout layers
dropout = nn.Dropout(p=0.5) # Adjust the dropout rate as needed

class EnsemModel(nn.Module):
  def __init__(self, roberta_model, xlmroberta_model, pooler, layer_norm, dropout):
    super(EnsemModel, self).__init__()
    self.roberta_model = roberta_model
    self.xlmroberta_model =xlmroberta_model
    self.pooler = pooler
    self.layer_norm = layer_norm
    self.dropout = dropout
    self.relu = nn.ReLU()
    self.classification_layer = nn.Sequential(nn.Linear(2048, 1))  # 1 dim for binary classification #1536 for small #2048 for large-checkpoint


  def forward(self, input_ids, attention_mask=None):

    # get the outputs of the models
    roberta_outputs = self.roberta_model(input_ids, attention_mask=attention_mask)
    xlmroberta_outputs = self.xlmroberta_model(input_ids, attention_mask=attention_mask)

    # apply pooler output
    # require permute since original shape of outputs from roberta models is (batch_size, sequence_length, hidden_size), 
    # for pooler we require it as (batch_size, hidden_size, seq_length) as seen in pytorch docu where C is the hidden size/channels and N is batch size
    roberta_pooled = self.pooler(roberta_outputs.last_hidden_state.permute(0,2,1))
    xlmroberta_pooled = self.pooler(xlmroberta_outputs.last_hidden_state.permute(0,2,1))

    # apply relu
    roberta_relu = self.relu(roberta_pooled.squeeze(-1))
    xlmroberta_relu = self.relu(xlmroberta_pooled.squeeze(-1))

    # apply layer normalization
    roberta_normalized = self.layer_norm(roberta_relu)
    xlmroberta_normalized = self.layer_norm(xlmroberta_relu)

    # apply multi-sample dropout x4
    roberta_dropped = self.dropout(roberta_normalized)
    xlmroberta_dropped = self.dropout(xlmroberta_normalized)

    roberta_dropped2 = self.dropout(roberta_dropped)
    xlmroberta_dropped2 = self.dropout(xlmroberta_dropped)

    roberta_dropped3 = self.dropout(roberta_dropped2)
    xlmroberta_dropped3 = self.dropout(xlmroberta_dropped2)

    roberta_dropped4 = self.dropout(roberta_dropped3)
    xlmroberta_dropped4 = self.dropout(xlmroberta_dropped3)

    # concatenate the outputs
    ensemble_output = t.cat((roberta_dropped4, xlmroberta_dropped4), dim=1) 

    # last layer
    output = self.classification_layer(ensemble_output)

    return output

# MODEL setup
model = EnsemModel(roberta_model, xlmroberta_model, pooler, layer_norm, dropout)
model.to(device)

# Define the optimizer and loss function
lr = 1e-05 
optimizer = t.optim.AdamW(model.parameters(),
                          lr=lr,
                          weight_decay= 0.004, 
                          betas = ([0.9,0.999]),
                          eps=1e-07)

criterion = nn.BCEWithLogitsLoss()

# training
for epoch in range(4):  # Modify the number of epochs as necessary
  model.train()
  running_loss = 0.0
  print(epoch)
  for i, (data, labels) in enumerate(train_loader, 0):
      print(i)
      input_ids = data['input_ids'].squeeze(dim=1).to(device) # need squeeze dim is wrong now
      attention_mask = data['attention_mask'].to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(input_ids=input_ids, attention_mask=attention_mask) 
      loss = criterion(outputs.view(-1), labels.type_as(outputs))
      loss.backward()
      optimizer.step()
      running_loss +=loss.item()

      # print loss value every 120 iterations and reset running loss
      if i % 120 == 119: 
        print('[%d, %5d] loss: %.3f' %
          (epoch, i, running_loss / 120)) 
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

      val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)
      val_outputs = t.sigmoid(val_outputs.detach())

      val_predictions = (val_outputs > 0.5).float()
      predictions.extend(val_predictions.cpu().detach().numpy())
      true_labels.extend(val_labels.cpu().detach().numpy())

    val_f1 = f1_score(true_labels, predictions)
    print(f'Validation F1 Score: {val_f1}')
