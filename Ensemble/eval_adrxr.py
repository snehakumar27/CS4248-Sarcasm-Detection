#################################################################
## TEST SCRIPT ADRXR                                           ##
## Soft Vote Ensemble approach of 3 models:                    ##
## 1. RobXLMRob (Customised Ensemble of RoBERTa and XLMRoBERTa ##
## 2. DistilBERT                                               ##
## 3. Albert                                                   ##
########## Prepared by: Matthias Koh Yong An ####################
#################################################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler, TensorDataset

from transformers import AutoTokenizer, AlbertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, DistilBertTokenizer, AdamW, RobertaModel, XLMRobertaModel, AlbertModel, DistilBertModel


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

model_path = "ensem3_model_2_complete.pt" #change based on checkpoint
X_test_directory = "../Datasets/clean_data/X_test.csv"  #"sent_test.csv"    #"X_test.csv"
y_test_directory = "../Datasets/clean_data/y_test.csv"  #"sent_test.csv"    #"y_test.csv"

X_test = pd.read_csv(X_test_directory)
y_test = pd.read_csv(y_test_directory)

## Prepare Validation & Test Data 
max_len = 60
batch_size = 48

model_name = 'roberta-large' #'roberta-base'
tokenizer =  RobertaTokenizer.from_pretrained(model_name)

model_name2 = 'distilbert-base-uncased' 
tokenizer2 =  DistilBertTokenizer.from_pretrained(model_name2)

model_name3 = 'albert-base-v2'
tokenizer3 =  AlbertTokenizer.from_pretrained(model_name3)

X_test = list(X_test['text']) 
y_test = list(y_test['label']) 

test_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_test, y_test)]
test_dataset2 = [(tokenizer2.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_test, y_test)]
test_dataset3 = [(tokenizer3.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_test, y_test)]

test_loader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
test_loader2 = t.utils.data.DataLoader(test_dataset2, batch_size=batch_size,shuffle=False)
test_loader3 = t.utils.data.DataLoader(test_dataset3, batch_size=batch_size,shuffle=False)

# load the pre-trained models
roberta_model = RobertaModel.from_pretrained("roberta-large") #roberta-base
xlmroberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-large") #xlm-roberta-base
distilbert_model= DistilBertModel.from_pretrained("distilbert-base-uncased")
albert_model = AlbertModel.from_pretrained("albert-base-v2")

## Init Model Class and Load Model
device = t.device("cuda") if t.cuda.is_available() else "cpu"
checkpoint = t.load(model_path, map_location=device)

# hyperparams global
dropout_value = 0.4

# define pooler output
pooler = nn.AdaptiveAvgPool1d(1) # output of size 1

# define multi-sample dropout layers
dropout = nn.Dropout(p=dropout_value) # adjust the dropout rate as needed

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
    distilbert_outputs = self.distilbert_model(input_ids2,attention_mask=attention_mask2)
    albert_outputs = self.albert_model(input_ids3,token_type_ids=token_type_ids, attention_mask=attention_mask3)

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
  

model = EnsemModel(roberta_model, xlmroberta_model, distilbert_model, albert_model, pooler, dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)



## Model Evaluation
model.eval()

test_f1 = 0.0
test_f12 = 0.0
test_f13 = 0.0
entest_f1 = 0.0

with t.no_grad():
    for batch, ((test_data, test_labels), (test_data2, test_labels2),(test_data3, test_labels3)) in enumerate(zip(test_loader, test_loader2, test_loader3),0):

        # valid input
        #### 1. RobXLMRob  ####
        test_input_ids = test_data['input_ids'].squeeze(dim=1).to(device)
        test_attention_mask = test_data['attention_mask'].to(device)
        test_labels = test_labels.to(device)

        #### 2. DistilBERT ####
        test_input_ids2 = test_data2['input_ids'].squeeze(dim=1).to(device)
        test_attention_mask2 = test_data2['attention_mask'].to(device)
        test_labels2 = test_labels2.to(device)

        #### 3. Albert ####
        test_input_ids3 = test_data3['input_ids'].squeeze(dim=1).to(device)
        test_token_type_ids = test_data3["token_type_ids"].squeeze(dim=1).to(device)
        test_attention_mask3 = test_data3['attention_mask'].squeeze(dim=1).to(device)
        test_labels3 = test_labels3.to(device)

        # place into model
        test_outputs, test_outputs2,test_outputs3 = model(input_ids=test_input_ids, attention_mask=test_attention_mask,input_ids2=test_input_ids2, attention_mask2=test_attention_mask2, input_ids3=test_input_ids3,token_type_ids=test_token_type_ids, attention_mask3=test_attention_mask3)


        f1_batch = f1_score(test_labels.cpu().numpy(), np.round(t.sigmoid(test_outputs).cpu().numpy()), average = 'binary')
        test_f1 += f1_batch

        f1_batch2 = f1_score(test_labels2.cpu().numpy(), np.round(t.sigmoid(test_outputs2).cpu().numpy()), average = 'binary')
        test_f12 += f1_batch2

        f1_batch3 = f1_score(test_labels3.cpu().numpy(), np.round(t.sigmoid(test_outputs3).cpu().numpy()), average = 'binary')
        test_f13 += f1_batch3

        #### EnsemModel Validation ###
        test_outputs = t.sigmoid(test_outputs.detach())
        test_outputs2 = t.sigmoid(test_outputs2.detach())
        test_outputs3 = t.sigmoid(test_outputs3.detach())

        # equal weights (can change weights as long sum to 1)
        test_eoutput =  0.15*test_outputs3 + 0.15*test_outputs2 + 0.70*test_outputs

        test_predictions = (test_eoutput > 0.42).float()

        enf1_batch = f1_score(test_labels.cpu().numpy(), test_predictions.cpu().detach().numpy(), average = 'binary')
        entest_f1 += enf1_batch
    
    num = len(test_loader)
    avg_test_f1 = test_f1 / num
    print("Test F1 robxlmrob: {:.4f}".format(avg_test_f1))

    avg_test_f12 = test_f12 / num
    print("Test F1 distil: {:.4f}".format(avg_test_f12))

    avg_test_f13 = test_f13 / num
    print("Test F1 albert: {:.4f}".format(avg_test_f13))

    enavg_test_f1 = entest_f1 / num
    print("Test F1 ensem_model: {:.4f}".format(enavg_test_f1))

