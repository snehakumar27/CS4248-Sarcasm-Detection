#################################################################
## TRAINING SCRIPT ADRXR                                       ##
## Soft Vote Ensemble approach of 3 models:                    ##
## 1. RobXLMRob (Customised Ensemble of RoBERTa and XLMRoBERTa ##
## 2. DistilBERT                                               ##
## 3. Albert                                                   ##
########## Prepared by: Matthias Koh Yong An ####################
#################################################################

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler

from transformers import AutoTokenizer, AlbertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, DistilBertTokenizer, AdamW, RobertaModel, XLMRobertaModel, AlbertModel, DistilBertModel

###############
## READ DATA ##
###############
# read in data
X_train_valid_directory = "../Datasets/clean_data/X_train_dev.csv"
y_train_valid_directory = "../Datasets/clean_data/y_train_dev.csv"
X_train_dev = pd.read_csv(X_train_valid_directory)
y_train_dev = pd.read_csv(y_train_valid_directory)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_dev['text'], y_train_dev['label'], test_size=0.11, random_state=4248)


##############
## TOKENIZE ##
##############
print('Tokenize started')
# tokenize the dataset and place into loaders
# hyperparams global
max_len = 60
batch_size = 48

#### 1. RobXLMRob ####
model_name = 'roberta-large' #'roberta-base'
tokenizer =  RobertaTokenizer.from_pretrained(model_name)

# local hyperparam
# max_len = 50 #if pref specify diff for diff models 
# batch_size = 8 #4

train_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset = [(tokenizer.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = t.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

#### 2. DistilBERT ####
model_name2 = 'distilbert-base-uncased' 
tokenizer2 =  DistilBertTokenizer.from_pretrained(model_name2)

# local hyperparam
# max_len = 50 #if pref specify diff for diff models 
# batch_size = 8 #4

train_dataset2 = [(tokenizer2.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset2 = [(tokenizer2.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader2 = t.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
valid_loader2 = t.utils.data.DataLoader(valid_dataset2, batch_size=batch_size)

#### 3. Albert ####
model_name3 = 'albert-base-v2' #'xlm-roberta-large' #'roberta-large' #roberta-base'
tokenizer3 =  AlbertTokenizer.from_pretrained(model_name3)

# local hyperparam
# max_len = 50 #if pref specify diff for diff models 
# batch_size = 8 #4

train_dataset3 = [(tokenizer3.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_train, y_train)] #truncation=True refers to the process of truncating sequences to a certain maximum length.
valid_dataset3 = [(tokenizer3.encode_plus(text, max_length=max_len, padding='max_length',truncation=True, return_tensors="pt"), label) for text, label in zip(X_valid, y_valid)]

train_loader3 = t.utils.data.DataLoader(train_dataset3, batch_size=batch_size, shuffle=True)
valid_loader3 = t.utils.data.DataLoader(valid_dataset3, batch_size=batch_size)

print('Tokenize completed')

#######################
## PRETRAINED MODELS ##
#######################

# load the pre-trained models
roberta_model = RobertaModel.from_pretrained("roberta-large") #roberta-base
xlmroberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-large") #xlm-roberta-base
distilbert_model= DistilBertModel.from_pretrained("distilbert-base-uncased")
albert_model = AlbertModel.from_pretrained("albert-base-v2")

####################################
## Ensemble Model Training Layers ##
####################################

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
  
#######################
## Initlialise Model ##
#######################

# usage of GPU
device = t.device("cuda") if t.cuda.is_available() else "cpu"

# initliatise model
model = EnsemModel(roberta_model, xlmroberta_model, distilbert_model, albert_model, pooler, dropout)
model.to(device)

# define the optimizer
# hyperparams
lr = 1e-05 #0.001
wd = 0.004
b1, b2 = 0.9, 0.999
eps = 1e-07
optimizer = t.optim.AdamW(model.parameters(),
                          lr=lr,
                          weight_decay= wd, 
                          betas = ([b1,b2]),
                          eps=eps)

# define loss
criterion = nn.BCEWithLogitsLoss()

####################################
##### Train Model + Validation #####
####################################
print('Training started')
# hyperparams
epoch_num = 5

# checkpointing
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 0,
    'step':0,
    'train_losses': [],
    'train_f1_scores': [],
    'train_losses2': [],
    'train_f12_scores': [],
    'train_losses3': [],
    'train_f13_scores': [],
    'valid_losses': [],
    'valid_f1_scores': [],
    'valid_losses2': [],
    'valid_f12_scores': [],
    'valid_losses3': [],
    'valid_f13_scores': [],
    'envalid_losses': [],
    'envalid_f1_scores': []
}


#### Train Model
for epoch in range(epoch_num):  # Modify the number of epochs as necessary
  model.train()

  running_loss = 0.0
  running_f1 = 0.0

  running_loss2 = 0.0
  running_f12 = 0.0

  running_loss3 = 0.0
  running_f13 = 0.0

  print(epoch)
  for step, ((data, labels), (data2, labels2),(data3, labels3)) in enumerate(zip(train_loader, train_loader2, train_loader3), 0):
      
      #print(i)

      # inputs
      #### 1. RobXLMRob  ####
      input_ids = data['input_ids'].squeeze(dim=1).to(device) # need squeeze dim is wrong now
      attention_mask = data['attention_mask'].to(device)
      labels = labels.to(device)

       #### 2. DistilBERT ####
      input_ids2 = data2['input_ids'].squeeze(dim=1).to(device) # need squeeze dim is wrong now
      attention_mask2 = data2['attention_mask'].to(device)
      labels2 = labels2.to(device)

      #### 3. Albert ####
      input_ids3 = data3['input_ids'].squeeze(dim=1).to(device) # need squeeze dim is wrong now
      token_type_ids = data3["token_type_ids"].squeeze(dim=1).to(device)
      attention_mask3 = data3['attention_mask'].squeeze(dim=1).to(device)
      labels3 = labels3.to(device)
      optimizer.zero_grad()

      # place into model
      outputs, outputs2, outputs3 = model(input_ids=input_ids, attention_mask=attention_mask,input_ids2=input_ids2, attention_mask2=attention_mask2,input_ids3=input_ids3, token_type_ids=token_type_ids, attention_mask3=attention_mask3)

      ############
      ### LOSS ###
      #### 1. RobXLMRob  ####
      loss = criterion(outputs.view(-1), labels.view(-1).type_as(outputs)) 
      f1_batch = f1_score(labels.cpu().numpy(), np.round(t.sigmoid(outputs.detach()).cpu().numpy()), average = 'binary')
      running_f1 += f1_batch
      loss.backward()

      #### 2. DistilBERT ####
      loss2 = criterion(outputs2.view(-1), labels2.view(-1).type_as(outputs2)) 
      f1_batch2 = f1_score(labels2.cpu().numpy(), np.round(t.sigmoid(outputs2.detach()).cpu().numpy()), average = 'binary')
      running_f12 += f1_batch2
      loss2.backward()

      #### 3. Albert ####
      loss3 = criterion(outputs3.view(-1), labels3.view(-1).type_as(outputs3))
      f1_batch3 = f1_score(labels3.cpu().numpy(), np.round(t.sigmoid(outputs3.detach()).cpu().numpy()), average = 'binary')
      running_f13 += f1_batch3
      loss3.backward()

      optimizer.step()

      # compound loss
      running_loss +=loss.item()
      running_loss2 +=loss2.item()
      running_loss3 +=loss3.item()

      # print loss value every 120 iterations and reset running loss
      if step % 120 == 119: 
        print("Train step results")
        #### 1. RobXLMRob  ####
        avg_loss = running_loss/120
        avg_f1 = running_f1/120
        print('[%d, %5d] robxlmrob avg_loss: %.4f avg_f1 %.4f' %
          (epoch + 1, step + 1, avg_loss, avg_f1)) 
        
        checkpoint['train_losses'].append(avg_loss)
        checkpoint['train_f1_scores'].append(avg_f1)
        
        #### 2. DistilBERT ####
        avg_loss2 = running_loss2/120
        avg_f12 = running_f12/120
        print('[%d, %5d] distil avg_loss2: %.4f avg_f12 %.4f' %
          (epoch + 1, step + 1, avg_loss2, avg_f12)) 
        
        checkpoint['train_losses2'].append(avg_loss2)
        checkpoint['train_f12_scores'].append(avg_f12)
        
        #### 3. Albert ####
        avg_loss3 = running_loss3/120
        avg_f13 = running_f13/120
        print('[%d, %5d] albert avg_loss3: %.4f avg_f13 %.4f' %
          (epoch + 1, step + 1, avg_loss3, avg_f13)) 
        checkpoint['train_losses3'].append(avg_loss3)
        checkpoint['train_f13_scores'].append(avg_f13)
        
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['step'] = step + 1

        t.save(checkpoint, f'ensem00_model_{epoch + 1}.pt')
        # update back to zero
        running_loss = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_f1 = 0.0
        running_f12 = 0.0
        running_f13 = 0.0

        #### Validation every 120 steps
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0

        val_loss2 = 0.0
        val_f12 = 0.0

        val_loss3 = 0.0
        val_f13 = 0.0

        enval_loss = 0.0
        enval_f1 = 0.0
        with t.no_grad():
            
            for (val_data, val_labels), (val_data2, val_labels2),(val_data3, val_labels3) in zip(valid_loader, valid_loader2, valid_loader3):
                
                # valid input
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
            
                ##################
                ### Valid LOSS ###
                #### 1. RobXLMRob  ####
                loss = criterion(val_outputs.view(-1), val_labels.view(-1).type_as(val_outputs)) 
                val_loss += loss.item()

                f1_batch = f1_score(val_labels.cpu().numpy(), np.round(t.sigmoid(val_outputs).cpu().numpy()), average = 'binary')
                val_f1 += f1_batch

                #### 2. DistilBERT ####
                loss2 = criterion(val_outputs2.view(-1), val_labels2.view(-1).type_as(val_outputs2)) 
                val_loss2 += loss2.item()

                f1_batch2 = f1_score(val_labels2.cpu().numpy(), np.round(t.sigmoid(val_outputs2).cpu().numpy()), average = 'binary')
                val_f12 += f1_batch2

                #### 3. Albert ####
                loss3 = criterion(val_outputs3.view(-1), val_labels3.view(-1).type_as(val_outputs3)) 
                val_loss3 += loss3.item()

                f1_batch3 = f1_score(val_labels3.cpu().numpy(), np.round(t.sigmoid(val_outputs3).cpu().numpy()), average = 'binary')
                val_f13 += f1_batch3
                
                #### EnsemModel Validation ###
                val_outputs = t.sigmoid(val_outputs.detach())
                val_outputs2 = t.sigmoid(val_outputs2.detach())
                val_outputs3 = t.sigmoid(val_outputs3.detach())

                # equal weights (can change to weightes sum to 1)
                val_eoutput =   0.15*val_outputs3 +0.15* val_outputs2 +0.70* val_outputs

                val_predictions = (val_eoutput > 0.42).float() #since not logits so i use this

                enf1_batch = f1_score(val_labels.cpu().numpy(), val_predictions.cpu().detach().numpy(), average = 'binary')
                enval_f1 += enf1_batch 
            
            #### 1. RobXLMRob  ####
            num = len(valid_loader)
            avg_val_loss = val_loss/num
            avg_val_f1 = val_f1/num

            print('[%d, %5d] robxlmrob avg_val_loss: %.4f avg_val_f1 %.4f' %
                  (epoch + 1, step + 1, avg_val_loss, avg_val_f1)) 
            checkpoint['valid_losses'].append(avg_val_loss)
            checkpoint['valid_f1_scores'].append(avg_val_f1)

            #### 2. DistilBERT ####
            avg_val_loss2 = val_loss2/num
            avg_val_f12 = val_f12/num

            print('[%d, %5d] distil avg_val_loss: %.4f avg_val_f1 %.4f' %
                  (epoch + 1, step + 1, avg_val_loss2, avg_val_f12)) 
            checkpoint['valid_losses2'].append(avg_val_loss2)
            checkpoint['valid_f12_scores'].append(avg_val_f12)

            #### 3. Albert ####
            avg_val_loss3 = val_loss3/num
            avg_val_f13 = val_f13/num

            print('[%d, %5d] albert avg_val_loss3: %.4f avg_val_f13 %.4f' %
                  (epoch + 1, step + 1, avg_val_loss3, avg_val_f13)) 
            checkpoint['valid_losses3'].append(avg_val_loss3)
            checkpoint['valid_f13_scores'].append(avg_val_f13)

            #### EnsemModel Validation ###
            enavg_val_f1 = enval_f1/num

            print('[%d, %5d] ensem  f1 %.4f' %
                  (epoch + 1, step + 1, enavg_val_f1))
            checkpoint['envalid_f1_scores'].append(enavg_val_f1)
            
        t.save(checkpoint, f'ensem00_model_{epoch + 1}.pt')


  checkpoint['model_state_dict'] = model.state_dict()
  checkpoint['optimizer_state_dict'] = optimizer.state_dict()
  checkpoint['epoch'] = epoch + 1
  t.save(checkpoint, f'ensem00_model_{epoch+1}_complete.pt')
