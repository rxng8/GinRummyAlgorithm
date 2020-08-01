# %%

# Import statement

import json

from typing import List, Dict

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use .to(<devvice>) to switch to gpu
# Use .cpu() to switch back to cpu

from tqdm import tqdm

import pickle


# %%


def __import_data_dense__(filename: str):
        # Read json or dat
    
    with open(filename, 'r') as f:
        obj = json.load(f)

    X_dense = np.empty(shape=(0, 208))
    Y_dense = np.empty(shape=(0, 52))

    for match in obj['X']:
        for turn in match:
            turn_vector = np.empty(shape=(0,))
            for feature in turn:
                for card in feature:
                    turn_vector = np.append(turn_vector, [card], axis=0)
            X_dense = np.append(X_dense, [turn_vector], axis=0)

    for match in obj['Y']:
        for turn in match:
            turn_vector = np.empty(shape=(0,))
            for card in turn:
                turn_vector = np.append(turn_vector, [card], axis=0)
            Y_dense = np.append(Y_dense, [turn_vector], axis=0)

    return X_dense, Y_dense


def __import_data_lstm__(filename: str):
    # Read json or dat
    
    with open(filename, 'r') as f:
        obj = json.load(f)

    # Set var
    match_list_X_f0 = []
    match_list_X_f1 = []
    match_list_X_f2 = []
    match_list_X_f3 = []

    match_list_Y = []

    for match in obj['X']:
        turn_list = np.empty(shape=(0,52))
        for turn in match:
            card_list = np.empty(shape=(0,))
            for card in turn[0]:
                card_list = np.append(card_list, [card], axis=0)
            turn_list = np.append(turn_list, [card_list], axis=0)
        match_list_X_f0.append(turn_list)

    for match in obj['X']:
        turn_list = np.empty(shape=(0,52))
        for turn in match:
            card_list = np.empty(shape=(0,))
            for card in turn[1]:
                card_list = np.append(card_list, [card], axis=0)
            turn_list = np.append(turn_list, [card_list], axis=0)
        match_list_X_f1.append(turn_list)

    for match in obj['X']:
        turn_list = np.empty(shape=(0,52))
        for turn in match:
            card_list = np.empty(shape=(0,))
            for card in turn[2]:
                card_list = np.append(card_list, [card], axis=0)
            turn_list = np.append(turn_list, [card_list], axis=0)
        match_list_X_f2.append(turn_list)

    for match in obj['X']:
        turn_list = np.empty(shape=(0,52))
        for turn in match:
            card_list = np.empty(shape=(0,))
            for card in turn[3]:
                card_list = np.append(card_list, [card], axis=0)
            turn_list = np.append(turn_list, [card_list], axis=0)
        match_list_X_f3.append(turn_list)

    for match in obj['Y']:
        turn_list = np.empty(shape=(0,52))
        for turn in match:
            card_list = np.empty(shape=(0,))
            for card in turn:
                card_list = np.append(card_list, [card], axis=0)
            turn_list = np.append(turn_list, [card_list], axis=0)
        match_list_Y.append(turn_list)

    return match_list_X_f0, match_list_X_f1, match_list_X_f2, match_list_X_f3, match_list_Y
    
def one_generator(x0, x1, x2, x3, y, n_match, seed=1):
    
    r = 0
    
    while True:

        r = r % n_match

        x_train_1 = np.reshape(x0[r], (1, x0[r].shape[0], x0[r].shape[1]))
        x_train_2 = np.reshape(x1[r], (1, x1[r].shape[0], x1[r].shape[1]))
        x_train_3 = np.reshape(x2[r], (1, x2[r].shape[0], x2[r].shape[1]))
        x_train_4 = np.reshape(x3[r], (1, x3[r].shape[0], x3[r].shape[1]))

        # Fix error in data
        if x_train_1.shape[1] == 0:
            r += seed
            continue

        y_train = np.reshape(y[r][y[r].shape[0] - 1], (1, y[r].shape[1]))

        assert x_train_1.shape == x_train_2.shape, "wrong data form"
        assert x_train_2.shape == x_train_3.shape, "wrong data form"
        assert x_train_3.shape == x_train_4.shape, "wrong data form"

        # assert x_train_1.shape == y_train.shape, "so wrong data form"
        r += seed
        yield [x_train_1, x_train_2, x_train_3, x_train_4], y_train

# %%

class Net(nn.Module):
    def __init__(self, input_shape: tuple):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.linear1 = nn.Linear(input_shape[1], 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 52)

    def forward(self, x: Tensor) -> Tensor:
        layer1 = F.relu(self.linear1(x.view(-1, self.input_shape[1])))
        layer2 = F.relu(self.linear2(layer1))
        layer3 = F.relu(self.linear3(layer2))
        layer4 = F.sigmoid(self.linear4(layer3))

        return layer4

class NetLSTM(n.Module):
    def __init__(self, shape):
        super().__init__()
        self.lstm00 = nn.LSTM(52, 64, 64)
        self.lstm01 = nn.LSTM(52, 64, 64)
        self.lstm02 = nn.LSTM(52, 64, 64)
        self.lstm03 = nn.LSTM(52, 64, 64)


    def forward(self, input:List):
        h0 = torch.randn(input[0].shape[0], 64, input[0].shape[1]).to(device)
        t0, h0 = self.lstm00(input[0], h0)
        h1 = torch.randn(input[1].shape[0], 64, input[1].shape[1]).to(device)
        t1, h1 = self.lstm01(input[1], h1)
        h2 = torch.randn(input[2].shape[0], 64, input[2].shape[1]).to(device)
        t2, h2 = self.lstm01(input[2], h2)
        h3 = torch.randn(input[3].shape[0], 64, input[3].shape[1]).to(device)
        t3, h3 = self.lstm01(input[3], h3)

        cc = torch.cat((t0, t1, t2, t3), axis=2)
        

        pass
# %%
X, Y = __import_data_dense__("./dataset/output_100.json")
# x0, x1, x2, x3, y = __import_data_lstm__("./dataset/output_100.json")

# %%

Y[1000]

# %%

x_train = torch.from_numpy(X).to(device).float()
y_train = torch.from_numpy(Y).to(device).float()

# %%

n_match = x_train.shape[0]

# %%

with torch.cuda.device(0):

    model = Net(x_train.shape)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    size = x_train.shape[0]
    batch_size = 1
    
    for _ in range(10):
        # for batch in tqdm(range(size - 1 // batch_size + 1), ncols=60):
        for batch in tqdm(range(n_match), ncols=60):
            output = model(x_train[batch * batch_size: (batch + 1) * batch_size])
            # loss = F.nll_loss(torch.log(output), y_train[batch * batch_size: (batch + 1) * batch_size])
            # The reason why the loss is high is because the actual label is not yet one-hot encoded!
            loss = nn.BCEWithLogitsLoss()(output, y_train[batch * batch_size: (batch + 1) * batch_size])
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Epoch:", _, "Loss:", loss.data)

# %%

y = model(x_train[2:3])
# %%

y[0]
# %%

y_train[400]