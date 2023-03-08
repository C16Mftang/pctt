import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import TemporalPC, myRNN
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'sine_wave_pc')
if not os.path.exists(result_path):
    os.makedirs(result_path)

seq_len = 200
amp = 5.
learn_iters = 100
learn_lr = 5e-4
inf_iters = 20
inf_iters_val = 20
inf_lr = 1e-3
input_size = 1
hidden_size = 36
output_size = 1
noise = 1.
print_interval = 2
train_size = 100
val_size = 10
test_size = 1
batch_size = 10
sparse_penal = 0.

# generate data
steps, y, y_trains, y_vals, y_tests = get_sine(seq_len, amp, train_size, val_size, test_size, noise, device)

# training with rnn
model = TemporalPC(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

train_losses = []
val_losses = []
for i in range(learn_iters):
    # accumulated loss for every batch
    train_loss = 0
    model.train()

    # batched training
    for j in range(0, train_size, batch_size):
        y_train = y_trains[j:j+batch_size]

        # we need to initialize hidden state every batch because otherwise 
        # the next batch will be considered as a continuation of the previous one
        h = model.init_hidden(batch_size).to(device)
        assert y_train.shape[0] == h.shape[0]
        y_pred = []
        for k in range(seq_len):
            # x: target, u: input
            optimizer.zero_grad()
            model.inference(inf_iters, inf_lr, y[:, k:k+1], y_train[:, k:k+1], h)
            model.update_grads(y[:, k:k+1], y_train[:, k:k+1], h)
            optimizer.step()
            h = model.z
            y_pred.append(model.pred_x)
        y_pred = torch.cat(y_pred, dim=1)
        loss = criterion(y_pred, y.repeat(batch_size, 1))
        train_loss += loss.item() / (train_size // batch_size)
    train_losses.append(train_loss)

    # test on validation data
    model.eval()
    h_val = model.init_hidden(val_size).to(device)
    y_pred_val = []
    for k in range(seq_len):
        model.inference(inf_iters_val, inf_lr, y[:, k:k+1], y_vals[:, k:k+1], h_val)
        h_val = model.z
        y_pred_val.append(model.pred_x)
    y_pred_val = torch.cat(y_pred_val, dim=1)
    val_loss = criterion(y_pred_val, y.repeat(val_size, 1)).item()
    val_losses.append(val_loss)

    if (i + 1) % print_interval == 0:
        print(
            f"Epoch [{i + 1}/{learn_iters}], "
            # f"Step [{i + 1}/{len(train_dataset)}], "
            f"Loss: {train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}"
        )


# perform inference on one example from both train and test set
# for supervised tasks there is no need to update the observation layer
# the PC model makes prediction just like an RNN does
def offline_inference(model, y_test):
    assert y_test.shape[0] == 1
    model.eval()
    h = model.init_hidden(1).to(device)
    y_pred = []
    for k in range(seq_len):
        model.inference(inf_iters_val, inf_lr, y[:, k:k+1], y_test[:, k:k+1], h)
        h = model.z
        y_pred.append(model.pred_x)
    y_pred = torch.cat(y_pred, dim=1)
    return y_pred

# generate predictions for both training and test data
y_pred = offline_inference(model, y_trains[0:1])
y_pred_test = offline_inference(model, y_tests[0:1])
        
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.savefig(result_path + f'/losses_len{seq_len}')

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(to_np(steps.squeeze()), to_np(y.squeeze()), label='true', c='k')
ax[0].plot(to_np(steps.squeeze()), to_np(y_pred[-1:].squeeze()), label='train pred')
ax[0].scatter(to_np(steps.squeeze()), to_np(y_trains[-1:].squeeze()), label='train data', facecolors='none', edgecolors='k')
ax[0].legend()
ax[0].set_title('Train')

ax[1].plot(to_np(steps.squeeze()), to_np(y.squeeze()), label='true', c='k')
ax[1].plot(to_np(steps.squeeze()), to_np(y_pred_test[-1:].squeeze()), label='test pred')
ax[1].scatter(to_np(steps.squeeze()), to_np(y_tests[-1:].squeeze()), label='test data', facecolors='none', edgecolors='k')
ax[1].legend()
ax[1].set_title('Test')
plt.savefig(result_path + f'/example_len{seq_len}')