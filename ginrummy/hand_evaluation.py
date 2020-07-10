# %%

# Import statement

import json

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

import pickle


class Net(nn.Module):
    def __init__(self, input_shape: tuple):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.linear1 = nn.Linear(input_shape[1], 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> Tensor:
        tensor = F.relu(self.linear1(x.view(-1, self.input_shape[1])))
        tensor = F.relu(self.linear2(tensor))
        tensor = F.relu(self.linear3(tensor))
        tensor = F.relu(self.linear4(tensor))
        tensor = F.relu(self.linear5(tensor))
        tensor = F.relu(self.linear6(tensor))
        out = F.sigmoid(self.linear7(tensor))

        return out

class SimpleNet(nn.Module):
    def __init__(self, input_shape: tuple):
        super(SimpleNet, self).__init__()
        self.input_shape = input_shape
        self.linear1 = nn.Linear(input_shape[1], 128)
        self.linear2 = nn.Linear(128, 52)
        self.linear3 = nn.Linear(52, 13)
        self.linear4 = nn.Linear(13, 1)

    def forward(self, x: Tensor) -> Tensor:
        tensor = F.relu(self.linear1(x.view(-1, self.input_shape[1])))
        tensor = F.relu(self.linear2(tensor))
        tensor = F.relu(self.linear3(tensor))
        out = F.sigmoid(self.linear4(tensor))

        return out

class Half_LSTM(nn.Module):
    def __init__(self, hand_shape: tuple, face_shape: tuple, hidden_size: int):
        super(Half_LSTM, self).__init__()
        self.linear_hand_face = nn.Linear(hand_shape[1], hidden_size)
        self.forget_gate = nn.Linear(hand_shape[1], hidden_size)
        self.input_gate = nn.Linear(hand_shape[1], hidden_size)
        self.candidate_gate = nn.Linear(hand_shape[1], hidden_size)

        # self.linear = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hand_shape[1], 1)


    def forward(self, hand: Tensor, face: Tensor) -> Tensor:
        
        assert hand.shape == face.shape, "Wrong data input size"
        
        hand_face = torch.add(hand, face)

        hand_face.mul_(self.c_gate(face))

        # hand_face = F.relu(self.linear_hand_face(hand_face))

        # forget_gate = F.sigmoid(self.forget_gate(hand))

        # input_gate = F.sigmoid(self.input_gate(hand))
        # candidate_gate = F.tanh(self.candidate_gate(hand))

        # hand_face.mul_(forget_gate).add_(torch.mul(input_gate, candidate_gate))
        
        tensor = F.sigmoid(self.linear(hand_face))

        return tensor

    # Input is the picked up vector (1D Tensor)
    def get_candidate_input(self, x: Tensor) -> Tensor:

        rank = torch.argmax(x) % 13
        suit = torch.argmax(x) // 13

        mask = x.new(x.size()).to(device).float()

        # Mask all suit
        for i in range(4):
            mask[i * 13 + rank] = 1
        
        # Mask all Json
        for i in range(13):
            mask[suit * 13 + i] = 1
        
        return mask

    # x: 2D Tensor (batch_size, face_up_feature)
    def c_gate (self, x: Tensor) -> Tensor:
        out = x.new(x.size()).to(device).float()
        for feature in out:
            feature = self.get_candidate_input(feature)
        return out
        
def import_csv(filename: str, separate_rate: float=0.75) -> ((Tensor, Tensor, Tensor, int), (Tensor, Tensor, Tensor, int), int):
    data = pd.read_csv(filename)
    x_df = data.iloc[:, :-1]
    y_df = data.iloc[:, -1:]

    separate_rate = separate_rate
    n_training = int(separate_rate * x_df.shape[0])

    x_train_hand = torch.from_numpy(x_df.iloc[:n_training,:52].values).to(device).float()
    x_train_face = torch.from_numpy(x_df.iloc[:n_training,52:].values).to(device).float()
    y_train = torch.from_numpy(y_df.iloc[:n_training,:].values).to(device).float()
    x_val_hand = torch.from_numpy(x_df.iloc[n_training:,:52].values).to(device).float()
    x_val_face = torch.from_numpy(x_df.iloc[n_training:,52:].values).to(device).float()
    y_val = torch.from_numpy(y_df.iloc[n_training:,:].values).to(device).float()
    n_match = data.shape[0]
    n_match_train = y_train.shape[0]
    n_match_val = y_val.shape[0]

    return (x_train_hand, x_train_face, y_train, n_match_train), (x_val_hand, x_val_face, y_val, n_match_val), n_match
# %%

# X, Y = __import_data_dense__("./dataset/output_100.json")
# x0, x1, x2, x3, y = __import_data_lstm__("./dataset/output_100.json")
(hand_train, face_train, y_train, n_train), (hand_val, face_val, y_val, n_val), n_match = \
    import_csv("./GinRummyMavenProject/data_picking.csv", 0.75)

# %%

with torch.cuda.device(0):

    # model = Half_LSTM(hand_train.shape, face_train.shape, 256)
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    size = n_match
    batch_size = 100
    
    for _ in range(30):
        # for batch in tqdm(range((size - 1) // batch_size + 1), ncols=60):
        # for batch in tqdm(range(n_match), ncols=60):
        for batch in tqdm(range(100), ncols=60):

            output = model(hand_train[batch * batch_size: (batch + 1) * batch_size], \
                face_train[batch * batch_size: (batch + 1) * batch_size])

            # loss = F.nll_loss(torch.log(output), y_train[batch * batch_size: (batch + 1) * batch_size])
            # The reason why the loss is high is because the actual label is not yet one-hot encoded!
            # training_loss = nn.BCEWithLogitsLoss()(output, y_train[batch * batch_size: (batch + 1) * batch_size])
            training_loss = nn.BCEWithLogitsLoss()(output, torch.randint(0,1,(batch_size,1)).to(device).float())
            model.zero_grad()
            training_loss.backward()
            optimizer.step()

        r = np.random.randint(0, n_val - batch_size)
        with torch.no_grad():
            val_output = model(hand_val[r: r+batch_size], \
                face_val[r: r+batch_size])
            val_loss = nn.BCEWithLogitsLoss()(val_output, y_val[r: r + batch_size])
        print("Epoch:", _, "Loss:", training_loss.data, "Val_loss:", val_loss.data)

# %%
r = np.random.randint(0, n_val)
y_pred = torch.round(model(hand_val[r:r+1], face_val[r:r+1]))
print(y_pred)
y_val[r:r+1]

# %%

percent_loss = 0
batch_predict_size = 400

r = np.random.randint(0, n_val - batch_predict_size)
y_pred = model(hand_val[r:r+batch_predict_size], face_val[r:r+batch_predict_size])
batch_val = y_val[r:r+batch_predict_size]

# %%
n_zero_predict = 0
n_zero_data = 0
count = 0
for i, result in enumerate(y_pred):
    predicted = torch.round(result[0])

    if predicted == 0:
        n_zero_predict += 1

    if y_val[i][0] == 0:
        n_zero_data += 1

    if  predicted== y_val[i][0]:
        count+=1

count / batch_predict_size

# %%

n_zero_predict





















