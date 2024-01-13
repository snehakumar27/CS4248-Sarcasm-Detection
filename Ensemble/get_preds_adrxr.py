#################################################################
## Obtain Predictions ADRXR                                    ##
## 1. RobXLMRob (Customised Ensemble of RoBERTa and XLMRoBERTa ##
## 2. DistilBERT                                               ##
## 3. Albert                                                   ##
################# Prepared by: Sneha Kumar ######################
#################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import torch as t
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, RobertaTokenizer, DistilBertTokenizer, RobertaModel, XLMRobertaModel, AlbertModel, DistilBertModel

from sklearn.model_selection import train_test_split

model_path = "ensem3_model_2_complete.pt"
X_train_valid_directory = "../../X_train_dev.csv"
y_train_valid_directory = "../../y_train_dev.csv"

X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)

"""# Prepare Data #
"""
max_len = 60
batch_size = 48 #4

#### 1. RobXLMRob ####
model_name = 'roberta-large' #'roberta-base'
tokenizer =  RobertaTokenizer.from_pretrained(model_name)

train_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = t.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

#### 2. DistilBERT ####
model_name2 = 'distilbert-base-uncased' 
tokenizer2 =  DistilBertTokenizer.from_pretrained(model_name2)

train_dataset2 = [(tokenizer2.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset2 = [(tokenizer2.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader2 = t.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=False)
valid_loader2 = t.utils.data.DataLoader(valid_dataset2, batch_size=batch_size, shuffle = False)

#### 3. Albert ####
model_name3 = 'albert-base-v2' #'xlm-roberta-large' #'roberta-large' #roberta-base'
tokenizer3 =  AlbertTokenizer.from_pretrained(model_name3)

train_dataset3 = [(tokenizer3.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset3 = [(tokenizer3.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader3 = t.utils.data.DataLoader(train_dataset3, batch_size=batch_size, shuffle=False)
valid_loader3 = t.utils.data.DataLoader(valid_dataset3, batch_size=batch_size, shuffle = False)



# load the pre-trained models
roberta_model = RobertaModel.from_pretrained("roberta-large") #roberta-base
xlmroberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-large") #xlm-roberta-base
distilbert_model= DistilBertModel.from_pretrained("distilbert-base-uncased")
albert_model = AlbertModel.from_pretrained("albert-base-v2")

# define pooler output
pooler = nn.AdaptiveAvgPool1d(1) # output of size 1

# define multi-sample dropout layers
dropout = nn.Dropout(p=0.4) # Adjust the dropout rate as needed

# define EnsemModel Class
class EnsemModel(nn.Module):
  def __init__(self, roberta_model, xlmroberta_model,distilbert_model, albert_model, pooler, dropout):
    super(EnsemModel, self).__init__()
    self.roberta_model = roberta_model
    self.xlmroberta_model = xlmroberta_model
    self.distilbert_model = distilbert_model
    self.albert_model = albert_model
    self.pooler = pooler
    self.relu = nn.ReLU()
    self.layer_norm = nn.LayerNorm(normalized_shape=1024, eps=1e-6) #if pretrain base 768 else large 1024
    self.layer_norm2 = nn.LayerNorm(normalized_shape=768, eps=1e-6)
    self.dropout = dropout
    self.classification_layer = nn.Sequential(nn.Linear(2048, 1)) # 1 dim for binary classification #if pretrain base 768x2=1536 else large 1024x2 = 2048
    self.classification_layer2 = nn.Sequential(nn.Linear(768, 1)) # utilise for albert and distilbert
    # Note: nn.Sigmoid() no need sigmoid layer since BCEwithLogitsLoss has a sigmoid layer too

  def forward(self, input_ids, attention_mask, input_ids2, attention_mask2, input_ids3, token_type_ids, attention_mask3):

    # get the outputs of the models
    roberta_outputs = self.roberta_model(input_ids, attention_mask=attention_mask)
    xlmroberta_outputs = self.xlmroberta_model(input_ids, attention_mask=attention_mask)
    distilbert_outputs = distilbert_model(input_ids2,attention_mask=attention_mask2)
    albert_outputs = albert_model(input_ids3,token_type_ids=token_type_ids, attention_mask=attention_mask3)

    # apply pooler output
    # require permute since original shape of outputs from roberta models is (batch_size, sequence_length, hidden_size),
    # for pooler we require it as (batch_size, hidden_size, seq_length) as seen in pytorch docu where C is the hidden size/channels and N is batch size
    roberta_pooled = self.pooler(roberta_outputs.last_hidden_state.permute(0,2,1))
    xlmroberta_pooled = self.pooler(xlmroberta_outputs.last_hidden_state.permute(0,2,1))
    distilbert_pooled = self.pooler(distilbert_outputs.last_hidden_state.permute(0,2,1))
    albert_pooled =self.pooler(albert_outputs.last_hidden_state.permute(0,2,1))

    # apply relu
    roberta_relu = self.relu(roberta_pooled.squeeze(-1))
    xlmroberta_relu = self.relu(xlmroberta_pooled.squeeze(-1))
    distilbert_relu = self.relu(distilbert_pooled.squeeze(-1))
    albert_relu = self.relu(albert_pooled.squeeze(-1))

    # apply layer normalization
    roberta_normalized = self.layer_norm(roberta_relu)
    xlmroberta_normalized = self.layer_norm(xlmroberta_relu)
    distilbert_normalized = self.layer_norm2(distilbert_relu)
    albert_normalized = self.layer_norm2(albert_relu)

    # apply multi-sample dropout
    roberta_dropped = self.dropout(roberta_normalized)
    xlmroberta_dropped = self.dropout(xlmroberta_normalized)
    distilbert_dropped = self.dropout(distilbert_normalized)
    albert_dropped = self.dropout(albert_normalized)

    roberta_dropped2 = self.dropout(roberta_dropped)
    xlmroberta_dropped2 = self.dropout(xlmroberta_dropped)
    distilbert_dropped2 = self.dropout(distilbert_dropped)
    albert_dropped2 = self.dropout(albert_dropped)

    roberta_dropped3 = self.dropout(roberta_dropped2)
    xlmroberta_dropped3 = self.dropout(xlmroberta_dropped2)
    distilbert_dropped3 = self.dropout(distilbert_dropped2)
    albert_dropped3 = self.dropout(albert_dropped2)

    roberta_dropped4 = self.dropout(roberta_dropped3)
    xlmroberta_dropped4 = self.dropout(xlmroberta_dropped3)
    distilbert_dropped4 = self.dropout(distilbert_dropped3)
    albert_dropped4 = self.dropout(albert_dropped3)

    # concatenate the outputs of RoBERTa and XLMRoBERTa
    ensemble_output = t.cat((roberta_dropped4, xlmroberta_dropped4), dim=1) 

    # last layer
    output = self.classification_layer(ensemble_output) #.detach()
    output2 = self.classification_layer2(distilbert_dropped4)
    output3 = self.classification_layer2(albert_dropped4)

    return output, output2, output3

#init and load model
device = t.device("cuda" if t.cuda.is_available() else "cpu")
checkpoint = t.load(model_path, map_location=device)
model = EnsemModel(roberta_model, xlmroberta_model,distilbert_model, albert_model, pooler, dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

#get predictions
model.eval()
rob_train_preds = []
rob_valid_preds = []
dis_train_preds = []
dis_valid_preds = []
alb_train_preds = []
alb_valid_preds = []
with t.no_grad():
    for (val_data, val_labels), (val_data2, val_labels2),(val_data3, val_labels3) in zip(valid_loader, valid_loader2, valid_loader3):
        #### 1. RobXLMRob  ####
        val_input_ids = val_data['input_ids'].squeeze(dim=1).to(device)
        val_attention_mask = val_data['attention_mask'].to(device)
        val_labels = val_labels.to(device)

        #### 2. DistilBERT ####
        val_input_ids2 = val_data2['input_ids'].squeeze(dim=1).to(device)
        val_attention_mask2 = val_data2['attention_mask'].to(device)
        val_labels2 = val_labels2.to(device)

        #### 3. Albert ####
        val_input_ids3 = val_data3['input_ids'].squeeze(dim=1).to(device)
        val_token_type_ids = val_data3["token_type_ids"].squeeze(dim=1).to(device)
        val_attention_mask3 = val_data3['attention_mask'].squeeze(dim=1).to(device)
        val_labels3 = val_labels3.to(device)

        # place into model
        val_outputs, val_outputs2,val_outputs3 = model(input_ids=val_input_ids, attention_mask=val_attention_mask,input_ids2=val_input_ids2, attention_mask2=val_attention_mask2, input_ids3=val_input_ids3,token_type_ids=val_token_type_ids, attention_mask3=val_attention_mask3)
        
        rob_valid_preds.extend(t.sigmoid(val_outputs).cpu().numpy())
        dis_valid_preds.extend(t.sigmoid(val_outputs2).cpu().numpy())
        alb_valid_preds.extend(t.sigmoid(val_outputs3).cpu().numpy())

    for (train_data, train_labels), (train_data2, train_labels2),(train_data3, train_labels3) in zip(train_loader, train_loader2, train_loader3):
       #### 1. RobXLMRob  ####
        train_input_ids = train_data['input_ids'].squeeze(dim=1).to(device)
        train_attention_mask = train_data['attention_mask'].to(device)
        train_labels = train_labels.to(device)

        #### 2. DistilBERT ####
        train_input_ids2 = train_data2['input_ids'].squeeze(dim=1).to(device)
        train_attention_mask2 = train_data2['attention_mask'].to(device)
        train_labels2 = train_labels2.to(device)

        #### 3. Albert ####
        train_input_ids3 = train_data3['input_ids'].squeeze(dim=1).to(device)
        train_token_type_ids = train_data3["token_type_ids"].squeeze(dim=1).to(device)
        train_attention_mask3 = train_data3['attention_mask'].squeeze(dim=1).to(device)
        train_labels3 = train_labels3.to(device)

        # place into model
        train_outputs, train_outputs2,train_outputs3 = model(input_ids=train_input_ids, attention_mask=train_attention_mask,input_ids2=train_input_ids2, attention_mask2=train_attention_mask2, input_ids3=train_input_ids3,token_type_ids=train_token_type_ids, attention_mask3=train_attention_mask3)
        
        rob_train_preds.extend(t.sigmoid(train_outputs).cpu().numpy())
        dis_train_preds.extend(t.sigmoid(train_outputs2).cpu().numpy())
        alb_train_preds.extend(t.sigmoid(train_outputs3).cpu().numpy())


    
rob_train_preds = np.array(rob_train_preds)
rob_valid_preds = np.array(rob_valid_preds)
dis_train_preds = np.array(dis_train_preds)
dis_valid_preds = np.array(dis_valid_preds)
alb_train_preds = np.array(alb_train_preds)
alb_valid_preds = np.array(alb_valid_preds)

np.save('robxlmrob_train_preds.npy', rob_train_preds)
np.save('robxlmrob_valid_preds.npy', rob_valid_preds)
np.save('distilbert_train_preds.npy', dis_train_preds)
np.save('distilbert_valid_preds.npy', dis_valid_preds)
np.save('albert_train_preds.npy', alb_train_preds)
np.save('albert_valid_preds.npy', alb_valid_preds)