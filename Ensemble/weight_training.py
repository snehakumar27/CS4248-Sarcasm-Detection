#################################################################
## Weight Optimization                                         ##
## 1. RobXLMRob (Customised Ensemble of RoBERTa and XLMRoBERTa ##
## 2. DistilBERT                                               ##
## 3. Albert                                                   ##
################# Prepared by: Sneha Kumar ######################
#################################################################

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import pickle

"""# Weighted Avg Optimization"""
# the following are the npy files output from the get preds py file
model1_train_preds = np.load('distilbert_train_preds.npy')    
model1_valid_preds = np.load('distilbert_valid_preds.npy')
model2_train_preds = np.load('albert_train_preds.npy')
model2_valid_preds = np.load('albert_valid_preds.npy')
model3_train_preds = np.load('robxlmrob_train_preds.npy')
model3_valid_preds = np.load('robxlmrob_valid_preds.npy')

# the following are true train and validation labels (located in google drive)
train_labels = np.load('train_labels.npy')
valid_labels = np.load('valid_labels.npy')

# Define batch size
batch_size = 128

# Split data into batches
train_batches = [range(i, min(i + batch_size, len(train_labels))) for i in range(0, len(train_labels), batch_size)]
valid_batches = [range(i, min(i + batch_size, len(valid_labels))) for i in range(0, len(valid_labels), batch_size)]


def train_loss(weight, batch_indices):
  batch_model1_train_preds = model1_train_preds[batch_indices]
  batch_model2_train_preds = model2_train_preds[batch_indices]
  batch_model3_train_preds = model3_train_preds[batch_indices]
  batch_train_labels = train_labels[batch_indices]

  pred_val = weight[0]*batch_model1_train_preds + weight[1]*batch_model2_train_preds + (1-weight[0]-weight[1])*batch_model3_train_preds
  log_probs = batch_train_labels * jnp.log(pred_val) + (1 - batch_train_labels) * jnp.log(1 - pred_val)
  return -jnp.mean(log_probs)

def val_loss(weight, batch_indices):
  batch_model1_valid_preds = model1_valid_preds[batch_indices]
  batch_model2_valid_preds = model2_valid_preds[batch_indices]
  batch_model3_valid_preds = model3_valid_preds[batch_indices]
  batch_valid_labels = valid_labels[batch_indices]

  pred_val = weight[0]*batch_model1_valid_preds + weight[1]*batch_model2_valid_preds + (1-weight[0]-weight[1])*batch_model3_valid_preds
  log_probs = batch_valid_labels * jnp.log(pred_val) + (1 - batch_valid_labels) * jnp.log(1 - pred_val)
  return -jnp.mean(log_probs)

# Perform gradient descent
w = jnp.array([1/3, 1/3])               #initialize with average
niter = 7                               #run for 7 iterations
lr = 0.0001                             #learning rate

#store training, validation loss as well as the values of c
tr_loss_traj = []
val_loss_traj = []
w_traj = []

for i in tqdm(range(niter)):
  train_loss_sum = 0
  for train_batch in train_batches:
    v, g = jax.value_and_grad(lambda w: train_loss(w, train_batch))(w)
    w = w - lr * g
    train_loss_sum += v
    w_traj.append(w)
  tr_loss_traj.append(train_loss_sum/len(train_batches))

  valid_loss_sum = 0
  for valid_batch in valid_batches:
    v = val_loss(w, valid_batch)
    valid_loss_sum += v
  val_loss_traj.append(valid_loss_sum/len(valid_batches))

checkpoint = {'loss': tr_loss_traj,
              'val_loss': val_loss_traj,
              'w': w_traj
              }

with open("weight_optim.pkl", "wb") as file:
    pickle.dump(checkpoint, file)


## for plotting trajectory 
import matplotlib.pyplot as plt

plt.plot(w_traj)
plt.plot(tr_loss_traj)
plt.plot(val_loss_traj)


## get F1 score with optimized weights: 
from sklearn.metrics import confusion_matrix

# Define batch size
batch_size = 128
opt_w = 0.5

# Calculate number of batches
num_batches = len(valid_labels) // batch_size

# counters
TP = 0
FP = 0
FN = 0

# Calculate F1 score for each batch
for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    batch_y_true = valid_labels[start:end]
    batch_y_model2_pred = model2_valid_preds[start:end]
    batch_y_model3_pred = model3_valid_preds[start:end]
    weighted_preds = np.round(opt_w*batch_y_model2_pred + (1-opt_w)*batch_y_model3_pred)
    tn, fp, fn, tp = confusion_matrix(batch_y_true, np.round(weighted_preds)).ravel()
    TP += tp
    FP += fp
    FN += fn

# Calculate aggregate F1 score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Aggregate F1 score: {f1}")