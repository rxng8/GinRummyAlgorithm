# %%

# Import statement

import json

from typing import List, Dict

import numpy as np
import pandas as pd

from keras.models import Model, Input
from keras.layers import LSTM, Dense, Concatenate
from keras.losses import categorical_crossentropy
from keras.activations import sigmoid, softmax, relu, tanh
from keras.metrics import Accuracy
from keras.optimizers import Adam




# %%

class HandEstimation:
    """ 
    This is the Hand estimaiton Data Preprocessing class.
    """
    def __init__(self, lr: int, n_epochs: int):
        """
        Initializing the data model.

        Args:
            lr (float): learning rate.
            n_epochs (int): number of epochs.
        """
        self.lr = lr
        self.n_epochs = n_epochs

    def __import_data_dense__(self, filename: str):
         # Read json or dat
        
        with open(filename, 'r') as f:
            obj = json.load(f)

        self.X_dense = np.empty(shape=(0, 208))
        self.Y_dense = np.empty(shape=(0, 52))

        for match in obj['X']:
            for turn in match:
                turn_vector = np.empty(shape=(0,))
                for feature in turn:
                    for card in feature:
                        turn_vector = np.append(turn_vector, [card], axis=0)
                self.X_dense = np.append(self.X_dense, [turn_vector], axis=0)

        for match in obj['Y']:
            for turn in match:
                turn_vector = np.empty(shape=(0,))
                for card in feature:
                    turn_vector = np.append(turn_vector, [card], axis=0)
                self.Y_dense = np.append(self.Y_dense, [turn_vector], axis=0)


    def __import_data_lstm__(self, filename: str):
        # Read json or dat
        
        with open(filename, 'r') as f:
            obj = json.load(f)

        # Set var
        match_list_X_f0 = np.array([[[]]])
        match_list_X_f1 = np.array([])
        match_list_X_f2 = np.array([])
        match_list_X_f3 = np.array([])

        match_list_Y = np.array([])

        for match in obj['X']:
            turn_list = np.array([[]])
            for turn in match:
                card_list = np.array([])
                for card in turn[0]:
                    card_list = np.append(card_list, [card], axis=0)
                turn_list = np.append(turn_list, [card_list], axis=0)
            match_list_X_f0 = np.append(match_list_X_f0, [turn_list], axis=0)

        print(match_list_X_f0.shape)

        print("Get here!")

        for match in obj['X']:
            turn_list = np.array([])
            for turn in match:
                card_list = np.array([])
                for card in turn[1]:
                     np.append(card_list, [card])
                np.append(turn_list, card_list)
            np.append(match_list_X_f1, turn_list)

        for match in obj['X']:
            turn_list = np.array([])
            for turn in match:
                card_list = np.array([])
                for card in turn[2]:
                     np.append(card_list, [card])
                np.append(turn_list, card_list)
            np.append(match_list_X_f2, turn_list)

        for match in obj['X']:
            turn_list = np.array([])
            for turn in match:
                card_list = np.array([])
                for card in turn[3]:
                    np.append(card_list, [card])
                np.append(turn_list, card_list)
            np.append(match_list_X_f3, turn_list)

        for match in obj['Y']:
            turn_list = np.array([])
            for turn in match:
                card_list = np.array([])
                for card in turn:
                    np.append(card_list, [card])
                np.append(turn_list, card_list)
            np.append(match_list_Y, turn_list)

        # self.match_list_X_f0 = np.array(match_list_X_f0)
        # self.match_list_X_f1 = np.array(match_list_X_f1)
        # self.match_list_X_f2 = np.array(match_list_X_f2)
        # self.match_list_X_f3 = np.array(match_list_X_f3)
        # self.match_list_Y = np.array(match_list_Y)

        self.match_list_X_f0 = match_list_X_f0
        self.match_list_X_f1 = match_list_X_f1
        self.match_list_X_f2 = match_list_X_f2
        self.match_list_X_f3 = match_list_X_f3
        self.match_list_Y = match_list_Y
        
    def dense_model(self):
        input = Input(shape=(208,), name = 'input')

        tensor = Dense(512, activation='relu', kernel_initializer='random_normal') (input)
        tensor = Dense(256, activation='relu', kernel_initializer='random_normal') (tensor)
        tensor = Dense(128, activation='relu', kernel_initializer='random_normal') (tensor)
        output = Dense(52, activation='sigmoid', kernel_initializer='random_normal') (tensor)

        model = Model(input, output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        return model

    def model (self, config=None):
        
        input_op_pick = Input(shape=(None, 52), name="op_pick")
        op_pick_lstm = LSTM(64, return_sequences=True, return_state=False) (input_op_pick)

        input_op_unpick = Input(shape=(None, 52), name="op_unpick")
        op_unpick_lstm = LSTM(64, return_sequences=True, return_state=False) (input_op_unpick)

        input_op_discard = Input(shape=(None, 52), name="op_discard")
        op_discard_lstm = LSTM(64, return_sequences=True, return_state=False) (input_op_discard)

        input_uncards = Input(shape=(None, 52), name="uncards")
        uncards_lstm = LSTM(64, return_sequences=True, return_state=False) (input_uncards)

        features = Concatenate(axis=2)([op_pick_lstm, op_unpick_lstm, op_discard_lstm, uncards_lstm])

        features = Dense(512, activation=relu) (features)

        features = Dense(128, activation=relu) (features)

        output = Dense(52, activation=sigmoid) (features)

        model = Model(inputs=[input_op_pick, input_op_unpick, input_op_discard, input_uncards], outputs=output)

        model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
        model.summary()

        return model



# %%


# Main

modelObj = HandEstimation(1e-3, 10)
modelObj.__import_data_dense__('./dataset/output.json')
model = modelObj.dense_model()

# %%

model.fit(x=[modelObj.match_list_X_f0,
            modelObj.match_list_X_f1,
            modelObj.match_list_X_f2,
            modelObj.match_list_X_f3],
            y= modelObj.match_list_Y)

# %%

model.fit(x=modelObj.X_dense[:1000],
        y=modelObj.Y_dense[:1000],
        batch_size=64,
        epochs=1000,
        verbose=1)
# %%

model.predict(modelObj.X_dense[1001:1010])

# %%

modelObj.Y_dense[1001:1010]

# %%

modelObj.X_dense.shape

# %%

model.save('simple_training.h5')