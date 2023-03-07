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

result_path = os.path.join('./results/', 'sine_wave')
if not os.path.exists(result_path):
    os.makedirs(result_path)

seq_len = 100
amp = 5.
learn_iters = 100
learn_lr = 1e-3
input_size = 1
hidden_size = 36
output_size = 1
noise = 1.
print_interval = 2
train_size = 100
val_size = 10
test_size = 1
batch_size = 10

# generate data
steps = torch.linspace(0., 4 * torch.pi, seq_len).reshape((1, seq_len))
y = amp * torch.sin(steps)
print(y.shape)

# generate training and validation data
y_trains = torch.zeros((train_size, seq_len)).to(device)
for s in range(train_size):
    torch.manual_seed(s)
    y_trains[s] = y + torch.randn_like(y) * noise

y_vals = torch.zeros((val_size, seq_len)).to(device)
for s in range(val_size):
    torch.manual_seed(s + train_size)
    y_vals[s] = y + torch.randn_like(y) * noise * 2

y_tests = torch.zeros((test_size, seq_len)).to(device)
for s in range(test_size):
    torch.manual_seed(s + train_size + val_size)
    y_tests[s] = y + torch.randn_like(y) * noise * 2

# training with rnn
model = myRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

train_losses = []
val_losses = []
for i in range(learn_iters):
    train_loss = 0
    # batched training
    for j in range(0, train_size, batch_size):
        y_train = y_trains[j:j+batch_size]
        h = model.init_hidden(batch_size).to(device)
        assert y_train.shape[0] == h.shape[0]
        y_pred = []
        for k in range(seq_len):
            h, output = model(y_train[:, k:k+1], h)
            y_pred.append(output)
        y_pred = torch.cat(y_pred, dim=1)
        loss = criterion(y_pred, y.repeat(batch_size, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / (train_size // batch_size)
    train_losses.append(train_loss)

    # test on validation data
    h_val = model.init_hidden(val_size).to(device)
    y_pred_val = []
    for k in range(seq_len):
        h_val, output = model(y_vals[:, k:k+1], h_val)
        y_pred_val.append(output)
    y_pred_val = torch.cat(y_pred_val, dim=1)
    loss_val = criterion(y_pred_val, y.repeat(val_size, 1))
    val_losses.append(loss_val.item())

    if (i + 1) % print_interval == 0:
        print(
            f"Epoch [{i + 1}/{learn_iters}], "
            # f"Step [{i + 1}/{len(train_dataset)}], "
            f"Loss: {loss.item():.4f}, "
            f"Val loss: {loss_val.item():.4f}"
        )

# there should be a unit for testing data here
h_test = model.init_hidden(test_size).to(device)
y_pred_test = []
for k in range(seq_len):
    h_test, output = model(y_tests[:, k:k+1], h_test)
    y_pred_test.append(output)
y_pred_test = torch.cat(y_pred_test, dim=1)
loss_test = criterion(y_pred_test, y.repeat(test_size, 1))

        
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.savefig(result_path + '/losses')

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
plt.savefig(result_path + '/example')