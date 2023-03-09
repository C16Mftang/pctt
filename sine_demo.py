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
seed = 48

# training and model configs
learn_iters = 100
learn_lr = 1e-3
update_step = 2
input_size = 1
hidden_size = 36
output_size = 1
step_update = False

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
model = myRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

logger_kwars = {'update_step': update_step, 'show': True}

trainer = RNNTrainer(
    seq_len=seq_len,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    step_update=step_update,
    logger_kwargs=logger_kwars,
    device=device
)

trainer.fit(train_loader, val_loader, learn_iters, batch_size)

train_losses = trainer.train_loss_
val_losses = trainer.val_loss_

# make predictions
train_pred = trainer.predict(data['X_train'], data['y_train'], train_size)
test_pred = trainer.predict(data['X_test'], data['y_test'], test_size)

# plotting loss
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.savefig(result_path + f'/losses_len{seq_len}')

# plotting predictions
steps = to_np(data['steps'].squeeze())
targ = to_np(data['y_train'][0])
train_data = to_np(data['X_train'][0])
test_data = to_np(data['X_test'][0])
train_pred = to_np(train_pred[0])
test_pred = to_np(test_pred[0])

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(steps, targ, label='target', c='k')
ax[0].plot(steps, train_pred, label='train pred')
ax[0].scatter(steps, train_data, label='train data', facecolors='none', edgecolors='k')
ax[0].legend()
ax[0].set_title('Train')
ax[0].set_yticks(np.arange(-10, 12.5, 2.5))

ax[1].plot(steps, targ, label='target', c='k')
ax[1].plot(steps, test_pred, label='test pred')
ax[1].scatter(steps, test_data, label='test data', facecolors='none', edgecolors='k')
ax[1].legend()
ax[1].set_title('Test')
ax[1].set_yticks(np.arange(-10, 12.5, 2.5))
plt.savefig(result_path + f'/example_len{seq_len}')