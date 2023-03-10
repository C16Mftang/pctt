import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import TemporalPC, myRNN
from src.utils import *
from src.get_data import *
from src.trainer import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'sine_wave_fitting')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# data configs
seq_len = 100
amp = 5.
noise = 1.
train_size = 10
val_size = 10
test_size = 1
batch_size = 10
seed = 1

# training and model configs
learn_iters = 100
learn_lr = 1e-3
update_step = 2
input_size = 1
hidden_size = 36
output_size = 1
step_update = False
inf_iters = 20
inf_lr = 1e-2

# load data
data = get_sine(seq_len, 
                amp, 
                train_size, 
                val_size, 
                test_size, 
                noise, 
                seed,
                device)

train_loader = DataLoader(DataWrapper(data['X_train'], data['y_train']), 
                          batch_size=batch_size)
val_loader = DataLoader(DataWrapper(data['X_val'], data['y_val']), 
                        batch_size=batch_size)

# model specifications
rnn = myRNN(input_size, hidden_size, output_size).to(device)
pcn = TemporalPC(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
rnn_opt = torch.optim.Adam(rnn.parameters(), lr=learn_lr)
pcn_opt = torch.optim.Adam(pcn.parameters(), lr=learn_lr)

logger_kwargs = {'update_step': update_step, 'show': True}
inference_kwargs = {'inf_iters': inf_iters, 'inf_lr': inf_lr}

rnn_trainer = RNNTrainer(
    seq_len=seq_len,
    model=rnn,
    criterion=criterion,
    optimizer=rnn_opt,
    step_update=step_update,
    logger_kwargs=logger_kwargs,
    device=device
)

pcn_trainer = PCTrainer(
    seq_len=seq_len,
    model=pcn,
    criterion=criterion,
    optimizer=pcn_opt,
    step_update=step_update,
    logger_kwargs=logger_kwargs,
    inference_kwargs=inference_kwargs,
    device=device
)

rnn_trainer.fit(train_loader, val_loader, learn_iters, batch_size)
rnn_train_losses = rnn_trainer.train_loss_
rnn_val_losses = rnn_trainer.val_loss_
# make predictions
rnn_train_pred = rnn_trainer.predict(data['X_train'], data['y_train'], train_size)
rnn_test_pred = rnn_trainer.predict(data['X_test'], data['y_test'], test_size)

pcn_trainer.fit(train_loader, val_loader, learn_iters, batch_size)
pcn_train_losses = pcn_trainer.train_loss_
pcn_val_losses = pcn_trainer.val_loss_
# make predictions
pcn_train_pred = pcn_trainer.predict(data['X_train'], data['y_train'], train_size)
pcn_test_pred = pcn_trainer.predict(data['X_test'], data['y_test'], test_size)

# plotting loss
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(rnn_train_losses, label='train')
ax[0].plot(rnn_val_losses, label='val')
ax[0].legend()
ax[0].set_title('RNN')
ax[1].plot(pcn_train_losses, label='train')
ax[1].plot(pcn_val_losses, label='val')
ax[1].legend()
ax[1].set_title('PCN')
plt.savefig(result_path + f'/losses_len{seq_len}_iters{learn_iters}_train{train_size}')

# plotting predictions
steps = to_np(data['steps'].squeeze())
targ = to_np(data['y_train'][0])
train_data = to_np(data['X_train'][0])
test_data = to_np(data['X_test'][0])
rnn_train_pred = to_np(rnn_train_pred[0])
rnn_test_pred = to_np(rnn_test_pred[0])
pcn_train_pred = to_np(pcn_train_pred[0])
pcn_test_pred = to_np(pcn_test_pred[0])

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(steps, targ, label='target', c='k')
ax[0].plot(steps, rnn_train_pred, label='rnn')
ax[0].plot(steps, pcn_train_pred, label='pcn')
ax[0].scatter(steps, train_data, label='train data', facecolors='none', edgecolors='k')
ax[0].legend()
ax[0].set_title('Train')
ax[0].set_yticks(np.arange(-10, 12.5, 2.5))

ax[1].plot(steps, targ, label='target', c='k')
ax[1].plot(steps, rnn_test_pred, label='rnn')
ax[1].plot(steps, pcn_test_pred, label='pcn')
ax[1].scatter(steps, test_data, label='test data', facecolors='none', edgecolors='k')
ax[1].legend()
ax[1].set_title('Test')
ax[1].set_yticks(np.arange(-10, 12.5, 2.5))
plt.savefig(result_path + f'/example_len{seq_len}_iters{learn_iters}_train{train_size}')