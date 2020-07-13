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

from keras.preprocessing.sequence import pad_sequences

import pickle


# %%

# CONSTANT

lr = 1e-3
n_epoch = 1

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
            self.X_dense = np.append(self.X_dense, [turn_vector], axis=0)

    for match in obj['Y']:
        for turn in match:
            turn_vector = np.empty(shape=(0,))
            for card in feature:
                turn_vector = np.append(turn_vector, [card], axis=0)
            self.Y_dense = np.append(self.Y_dense, [turn_vector], axis=0)

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
    
def __import_data_pad__(filename: str):
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
        turn_list = []
        for turn in match:
            card_list = []
            for card in turn[0]:
                card_list.append(card)
            turn_list.append(card_list)
        match_list_X_f0.append(turn_list)

    for match in obj['X']:
        turn_list = []
        for turn in match:
            card_list = []
            for card in turn[1]:
                card_list.append(card)
            turn_list.append(card_list)
        match_list_X_f1.append(turn_list)

    for match in obj['X']:
        turn_list = []
        for turn in match:
            card_list = []
            for card in turn[2]:
                card_list.append(card)
            turn_list.append(card_list)
        match_list_X_f2.append(turn_list)

    for match in obj['X']:
        turn_list = []
        for turn in match:
            card_list = []
            for card in turn[3]:
                card_list.append(card)
            turn_list.append(card_list)
        match_list_X_f3.append(turn_list)

    for match in obj['Y']:
        turn_list = []
        for turn in match:
            card_list = []
            for card in turn:
                card_list.append(card)
            turn_list.append(card_list)
        match_list_Y.append(turn_list)

    return match_list_X_f0, match_list_X_f1, match_list_X_f2, match_list_X_f3, match_list_Y

def data_lstm_generator(match_list_X_f0, match_list_X_f1, match_list_X_f2, match_list_X_f3, match_list_Y, batch_size: int, n_match: int):
    r = 0
    while True:
        
        if r + batch_size >= n_match:
            r = 0

        max_length = get_max_length(match_list_X_f0[r: r+batch_size])

        x_train_1 = np.empty(shape=(batch_size, max_length, match_list_X_f0[0].shape[1]))
        x_train_2 = np.empty(shape=(batch_size, max_length, match_list_X_f0[0].shape[1]))
        x_train_3 = np.empty(shape=(batch_size, max_length, match_list_X_f0[0].shape[1]))
        x_train_4 = np.empty(shape=(batch_size, max_length, match_list_X_f0[0].shape[1]))

        y_train = np.empty(shape=(batch_size, max_length, match_list_Y[0].shape[1]))

        for i in range(0, batch_size):
            x_train_1[i] = pad_data(np.reshape(match_list_X_f0[r+i], (1, *match_list_X_f0[r+i].shape)), max_length)
            x_train_2[i] = pad_data(np.reshape(match_list_X_f1[r+i], (1, *match_list_X_f1[r+i].shape)), max_length)
            x_train_3[i] = pad_data(np.reshape(match_list_X_f2[r+i], (1, *match_list_X_f2[r+i].shape)), max_length)
            x_train_4[i] = pad_data(np.reshape(match_list_X_f3[r+i], (1, *match_list_X_f3[r+i].shape)), max_length)

            y_train[i] = pad_data(np.reshape(match_list_Y[r+i], (1, *match_list_Y[r+i].shape)), max_length)

        # assert x_train_1.shape == x_train_2.shape, "wrong data form"
        # assert x_train_2.shape == x_train_3.shape, "wrong data form"
        # assert x_train_3.shape == x_train_4.shape, "wrong data form"
        r += batch_size

        yield [x_train_1, x_train_2, x_train_3, x_train_4], y_train

def get_max_length(x):
    max_length = -1
    for match in x:
        if match.shape[0] > max_length:
            max_length = match.shape[0]

    return max_length

def dense_model():
    input = Input(shape=(208,), name = 'input')

    tensor = Dense(512, activation='relu', kernel_initializer='random_normal') (input)
    tensor = Dense(256, activation='relu', kernel_initializer='random_normal') (tensor)
    tensor = Dense(128, activation='relu', kernel_initializer='random_normal') (tensor)
    output = Dense(52, activation='sigmoid', kernel_initializer='random_normal') (tensor)

    model = Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model

def lstm_model_simple (input_max_length=None, config=None):
    
    input_op_pick = Input(shape=(input_max_length, 52), name="op_pick")
    op_pick_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_pick)

    input_op_unpick = Input(shape=(input_max_length, 52), name="op_unpick")
    op_unpick_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_unpick)

    input_op_discard = Input(shape=(input_max_length, 52), name="op_discard")
    op_discard_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_discard)

    features = Concatenate(axis=1)([op_pick_lstm, op_unpick_lstm, op_discard_lstm])

    features = Dense(256, activation=relu, kernel_initializer='random_normal') (features)

    input_uncards = Input(shape=(input_max_length, 52), name="uncards")
    uncards_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_uncards)

    uncards_dense = Dense(96, activation=relu, kernel_initializer='random_normal') (uncards_lstm)

    features = Concatenate(axis=1)([features, uncards_dense])

    features = Dense(512, activation=relu, kernel_initializer='random_normal') (features)

    features = Dense(128, activation=relu, kernel_initializer='random_normal') (features)

    output = Dense(52, activation='sigmoid', kernel_initializer='random_normal') (features)

    model = Model(inputs=[input_op_pick, input_op_unpick, input_op_discard, input_uncards], outputs=output)

    # adam = Adam(learning_rate=0.01)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def lstm_model (input_max_length=None, config=None):
    
    input_op_pick = Input(shape=(input_max_length, 52), name="op_pick")
    op_pick_lstm = LSTM(128, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_pick)

    input_op_unpick = Input(shape=(input_max_length, 52), name="op_unpick")
    op_unpick_lstm = LSTM(128, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_unpick)

    input_op_discard = Input(shape=(input_max_length, 52), name="op_discard")
    op_discard_lstm = LSTM(128, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_discard)

    input_uncards = Input(shape=(input_max_length, 52), name="uncards")
    uncards_lstm = LSTM(128, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_uncards)

    features = Concatenate(axis=1)([op_pick_lstm, op_unpick_lstm, op_discard_lstm, uncards_lstm])

    features = Dense(768, activation=relu, kernel_initializer='random_normal') (features)

    features = Dense(256, activation=relu, kernel_initializer='random_normal') (features)

    features = Dense(64, activation=relu, kernel_initializer='random_normal') (features)

    output = Dense(52, activation='sigmoid', kernel_initializer='random_normal') (features)

    model = Model(inputs=[input_op_pick, input_op_unpick, input_op_discard, input_uncards], outputs=output)

    # adam = Adam(learning_rate=0.01)

    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def lstm_model_2 (input_max_length=None, config=None):
    
    input_op_pick = Input(shape=(input_max_length, 52), name="op_pick")
    op_pick_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_pick)

    input_op_unpick = Input(shape=(input_max_length, 52), name="op_unpick")
    op_unpick_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_unpick)

    input_op_discard = Input(shape=(input_max_length, 52), name="op_discard")
    op_discard_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_op_discard)

    features = Concatenate(axis=1)([op_pick_lstm, op_unpick_lstm, op_discard_lstm])

    input_uncards = Input(shape=(input_max_length, 52), name="uncards")
    uncards_lstm = LSTM(64, return_sequences=False, return_state=False, kernel_initializer='random_normal') (input_uncards)

    features = Concatenate(axis=1)([features, uncards_lstm])

    features = Dense(256, activation=relu, kernel_initializer='random_normal') (features)

    # features = Dense(256, activation=relu, kernel_initializer='random_normal') (features)

    # features = Dense(64, activation=relu, kernel_initializer='random_normal') (features)

    output = Dense(52, activation='sigmoid', kernel_initializer='random_normal') (features)

    model = Model(inputs=[input_op_pick, input_op_unpick, input_op_discard, input_uncards], outputs=output)

    # adam = Adam(learning_rate=0.01)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy'])
    model.summary()

    return model

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

def pad_data (x, max_length):
    y = x
    for i, turn in enumerate(x):
        if turn.shape[0] < max_length:
            zeros = np.zeros(shape=(max_length - turn.shape[0], turn.shape[1]))
            y[i] = np.append(y[i], zeros, axis=0)

    return y


# %%
x0, x1, x2, x3, y = __import_data_lstm__("./dataset/output_100.json")
n_match = len(x0)
# %%

# model = lstm_model_simple()

# model = lstm_model()
model = lstm_model_2()

# %%

model_b = lstm_model_2()
# %%


# %%

# model.fit(x=one_generator(x0, x1, x2, x3, y, n_match, 0)[0], y=one_generator(x0, x1, x2, x3, y, n_match, 0)[1], epochs=1, verbose=1)



history = model_b.fit_generator(one_generator(x0[:1500], x1[:1500], x2[:1500], x3[:1500], y[:1500], 1499), \
    steps_per_epoch=1499, \
    epochs=50,\
    verbose=1,\
    validation_data= one_generator(x0[500:], x1[500:], x2[500:], x3[500:], y[500:], 499),\
    validation_steps= 100,)


# %%

filename = 'lstm_simple_200'

with open('./history/' + filename + '_history.pkl', 'ab') as file_pi:
    pickle.dump(history.history, file_pi)

# np.save('./history/cdsc.npy', history.history)

# %%

with open('./history/' + filename + '_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# %%
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%

model.save('lstm_200_200epoch.h5')  # save everything in HDF5 format
model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open("lstm_200_200epoch_config.json", "w") as f:
    f.write(model_json)
model.save_weights('lstm_200_200epoch_weights.h5') # save just the weights.

# %%

############################################# Predict

# Make prediction data
def make_predict_data():
    pass

r = 2

ax0 = np.reshape(x0[r], (1, *x0[r].shape))
ax1 = np.reshape(x1[r], (1, *x1[r].shape))
ax2 = np.reshape(x2[r], (1, *x2[r].shape))
ax3 = np.reshape(x3[r], (1, *x3[r].shape))

ay = np.reshape(y[r], (1, *y[r].shape))


# ax0.shape
# len(y)

# %%


y_hat = model.predict([ax0[:,:5,:], ax1[:,:5,:], ax2[:,:5,:], ax3[:,:5,:]])
# %%
y_hat


# %%

ay[0][4]

# %%
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%

# summarize history for loss
plt.plot(model.history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
filename = "test"
model.save(filename + ".h5")  # save everything in HDF5 format

model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open(filename + "_config.json", "w") as f:
    f.write(model_json)

model.save_weights(filename + "_weights.h5") # save just the weights.


# %%


from tensorflow import keras
model = keras.models.load_model('./GinRummyMavenProject/src/main/java/model/lstm_200_150epoch.h5')

# %%

model.summary