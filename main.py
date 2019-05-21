import torch
import numpy as np
from LSTM import AttentionLSTM, DilatedRNN
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler


# SAMPLE CODE FOR LSTM, RESIDUAL LSTM AND ATTENTIVE LSTM TO PREDICT NEXT VALUE OF FIBONACCI SEQUENCE.

def fibo(n, a, b):
    fibo_list = [a, b]
    for i in range(n-2):
        new = fibo_list[-2] + fibo_list[-1]
        fibo_list.append(new)
    return fibo_list


fibo_len = 20
data = [fibo(fibo_len, random.randint(-20, 20), random.randint(-10, 10)) for _ in range(20)]
input_ = torch.FloatTensor(data).unsqueeze(1)

data_y = [fibo(fibo_len, random.randint(-1000, 1000), random.randint(-200, 2000))]
target = torch.FloatTensor([data_y[0][:-1]]).unsqueeze(1)
y_true = torch.FloatTensor([data_y[0][-1]]).unsqueeze(1)

print(input_.size())
print(target.size())
print(y_true.size())

model = AttentionLSTM(input_dim=fibo_len, driving_dim=20, encoder_dim=2, decoder_dim=5, encoder_type='LSTM',
                      decoder_type='ResidualLSTM')

iters = 250
learning_rate = 1e-2

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

cost_list = []
for iter in range(iters):
    batch_x = autograd.Variable(input_)
    batch_y = autograd.Variable(y_true)
    optimizer.zero_grad()
    pred = model.forward(input_=input_, target=target)
    cost = criterion(pred, batch_y.squeeze(0))
    cost_list.append(cost)
    cost.backward()
    optimizer.step()
    print('Iter ' + str(iter) + ': ' + str(cost.data.item()))

print(y_true)
print(pred)
plt.plot(range(len(cost_list)), cost_list)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# # SAMPLE CODE FOR DILATED LSTM TO PREDICT NEXT VALUE OF FIBONACCI SEQUENCE.
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def fibo(n, a, b):
#     fibo_list = [a, b]
#     for i in range(n - 2):
#         new = fibo_list[-2] + fibo_list[-1]
#         fibo_list.append(new)
#     return fibo_list
#
#
# data = [fibo(10, random.randint(-5, 5), random.randint(-2, 2)) for _ in range(3)]
# data_x = [np.array(item[:len(item) - 1]) for item in data]
# data_y = [np.array(item[-1]) for item in data]
#
# train_xx = data_x[:90]
# train_yy = data_y[:90]
# test_xx = data_x[90:]
# test_yy = data_y[90:]
# n_steps = len(data_x[0])
#
# input_dim = 1
# n_class = 1
# # model config
# layers_num = 5
# cells = ["ResidualLSTM"] * layers_num  # only support LSTM
# hidden_dims = [2] * layers_num  # Give a list of the dimension in each layer
# dilations = [1, 1, 2, 4, 4]  # Give a list of the dilation in each layer
# assert (len(hidden_dims) == len(dilations))
#
# # learning config
# batch_size = 1
# learning_rate = 1.0e-2
# training_iters = 2000
# display_step = 300
#
# print("Build prediction graph!!!")
# print("==> Building a dRNN with %s cells" % cells)
# model = DilatedRNN(hidden_dims, cells, dilations, n_class, input_dim)
# model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()
#
# cost_list = []
# for iter in range(training_iters):  # Number of Epoch. In this case, I set it to be 1, you can change more.
#     cost = 0
#     for step in range(len(train_xx)):
#         x = torch.from_numpy(train_xx[step]).view(input_dim, train_xx[step].shape[0]).type(torch.FloatTensor)
#         y = torch.from_numpy(train_yy[step]).view(1).type(torch.FloatTensor)
#         batch_x = autograd.Variable(x.view(-1, n_steps, input_dim))
#         batch_y = autograd.Variable(y)
#         # reshape inputs
#         x_reformat = batch_x.view(n_steps, batch_size,
#                                   input_dim)  # n_steps tensors (batch_size,input_dim)= (1,1) this case.
#         optimizer.zero_grad()
#
#         pred = model.forward(x_reformat)
#         cost_ = criterion(pred.squeeze(0), batch_y.type(torch.FloatTensor))
#         cost += cost_
#
#     cost = 1 / len(train_xx) * cost
#     cost_list.append(cost)
#
#     cost.backward()
#     optimizer.step()
#     print('Iter ' + str(iter) + ': ' + str(cost.data.item()))
#
# print("END!!!")
# plt.plot(range(len(cost_list)), cost_list)
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# # SAMPLE CODE FOR LSTM, RESIDUAL LSTM AND ATTENTIVE LSTM TO PREDICT NEXT VALUE OF RETAIL DATA.
#
# df = pd.read_csv('Retail.csv')
# scaler = StandardScaler()
# df = pd.DataFrame(scaler.fit_transform(df))
#
# input = [list(df.values[i]) for i in range(df.shape[0])]
#
# batch_size = 1000
# iters = 30
# learning_rate = 1e-2
#
# model = AttentionLSTM(input_dim=len(input[0]), driving_dim=batch_size-1, encoder_dim=10, decoder_dim=10,
#                       encoder_type='ResidualLSTM', decoder_type='LSTM')
#
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()
#
# costs = []
# for iter in range(iters):
#     print('Iter: ' + str(iter))
#     cost_list = []
#     random.shuffle(input)
#     batch_num = len(input) // batch_size
#
#     for batch_index in range(batch_num):
#         data = input[batch_index * batch_size:(batch_index + 1) * batch_size]
#         input_ = torch.FloatTensor([data[i] for i in range(len(data)-1)]).unsqueeze(1)
#         data_y = [data[-1]]
#         target = torch.FloatTensor([data_y[0][:-1]]).unsqueeze(1)
#         y_true = torch.FloatTensor([data_y[0][-1]]).unsqueeze(1)
#
#         batch_x = autograd.Variable(input_)
#         batch_y = autograd.Variable(y_true)
#
#         optimizer.zero_grad()
#         pred = model.forward(input_=input_, target=target)
#         cost = criterion(pred, batch_y.squeeze(0))
#         cost_list.append(cost.data.item())
#         print('Batch: ' + str(batch_index))
#         cost.backward()
#         optimizer.step()
#     costs.append(np.mean(cost_list))
#     print('Iter ' + str(iter) + ': ' + str(np.mean(cost_list)))
#
# plt.plot(range(len(costs)), costs)
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# # SAMPLE CODE FOR DILATED LSTM TO PREDICT NEXT VALUE OF RETAIL DATA.
#
# df = pd.read_csv('Retail.csv')
# scaler = StandardScaler()
# df = pd.DataFrame(scaler.fit_transform(df))
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# data = [df[df.index == i].values.tolist()[0] for i in range(1)]
# data_x = [np.array(item[:len(item) - 1]) for item in data]
# data_y = [np.array(item[-1]) for item in data]
#
# train_xx = data_x[:90]
# train_yy = data_y[:90]
# test_xx = data_x[90:]
# test_yy = data_y[90:]
# n_steps = len(data_x[0])
#
# input_dim = 1
# n_class = 1
# # model config
# layers_num = 5
# cells = ["ResidualLSTM"] * layers_num  # only support LSTM
# hidden_dims = [2] * layers_num  # Give a list of the dimension in each layer
# dilations = [1, 1, 2, 4, 4]  # Give a list of the dilation in each layer
# assert (len(hidden_dims) == len(dilations))
#
# # learning config
# batch_size = 1
# learning_rate = 1.0e-2
# training_iters = 200
# display_step = 300
#
# print("Build prediction graph!!!")
# print("==> Building a dRNN with %s cells" % cells)
# model = DilatedRNN(hidden_dims, cells, dilations, n_class, input_dim)
# model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()
#
# cost_list = []
# for iter in range(training_iters):  # Number of Epoch. In this case, I set it to be 1, you can change more.
#     cost = 0
#     for step in range(len(train_xx)):
#         x = torch.from_numpy(train_xx[step]).view(input_dim, train_xx[step].shape[0]).type(torch.FloatTensor)
#         y = torch.from_numpy(train_yy[step]).view(1).type(torch.FloatTensor)
#         batch_x = autograd.Variable(x.view(-1, n_steps, input_dim))
#         batch_y = autograd.Variable(y)
#         # reshape inputs
#         x_reformat = batch_x.view(n_steps, batch_size,
#                                   input_dim)  # n_steps tensors (batch_size,input_dim)= (1,1) this case.
#         optimizer.zero_grad()
#
#         pred = model.forward(x_reformat)
#         cost_ = criterion(pred.squeeze(0), batch_y.type(torch.FloatTensor))
#         cost += cost_
#
#     cost = 1 / len(train_xx) * cost
#     cost_list.append(cost)
#
#     cost.backward()
#     optimizer.step()
#     print('Iter ' + str(iter) + ': ' + str(cost.data.item()))
#
# print("END!!!")
# plt.plot(range(len(cost_list)), cost_list)
# plt.show()
