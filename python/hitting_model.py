# %%



# Import statement

import json

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

## Keras
import keras
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Concatenate
from keras.losses import categorical_crossentropy
from keras.activations import sigmoid, softmax, relu, tanh
from keras.metrics import Accuracy
from keras.optimizers import Adam

import keras.backend as K

import pickle



def import_csv(filename: str, separate_rate: float=0.75):
    data = pd.read_csv(filename)

    x_df = data.iloc[:, :-1]
    y_df = data.iloc[:, -1:]

    separate_rate = separate_rate
    n_training = int(separate_rate * x_df.shape[0])

    x_train = x_df.iloc[:n_training,:].values
    y_train = y_df.iloc[:n_training,:].values
    x_val = x_df.iloc[n_training:,:].values
    y_val = y_df.iloc[n_training:,:].values
    n_match = data.shape[0]
    n_match_train = y_train.shape[0]
    n_match_val = y_val.shape[0]

    return (x_train, y_train, n_match_train), (x_val, y_val, n_match_val), n_match

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model(input_shape):
    '''
    input: in
    output: out
    '''
    inp = Input(shape=input_shape)

    tensor = Dense(64, activation='relu', kernel_initializer='random_normal') (inp)

    tensor = Dense(32, activation='relu', kernel_initializer='random_normal') (tensor)

    output = Dense(1, activation='relu', kernel_initializer='random_normal')(tensor)
    
    model = Model(inp, output)
    model.compile(optimizer='adam', loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

    model.summary()
    return model

# %%

# (x_train_1, y_train_1, n_train_1), (x_val_1, y_val_1, n_val_1), n_match_1 = \
#     import_csv("./GinRummyMavenProject/data_knock_2_simple.csv", 0.95)

# (x_train_2, y_train_2, n_train_2), (x_val_2, y_val_2, n_val_2), n_match_2 = \
#     import_csv("./GinRummyMavenProject/data_knock_undercut.csv", 0.95)


# x_train = np.append(x_train_1, x_train_2, axis=0)
# y_train = np.append(y_train_1, y_train_2, axis=0)
# x_val = np.append(x_val_1, x_val_2, axis=0)
# y_val = np.append(y_val_1, y_val_2, axis=0)
# n_train = n_train_1 + n_train_2
# n_val = n_val_1 + n_val_2
# n_match = n_match_1 + n_match_2


# %%

(x_train, y_train, n_train), (x_val, y_val, n_val), n_match = \
    import_csv("../java/MavenProject/dataset/hit_sp_20000_v6.csv", 0.9)

# %%

# x_train.shape
x_train[:6]

# %%
input_shape = x_train[0].shape
model = build_model(input_shape)


# %%
size = y_train.shape[0]
batch_size = 100
history = model.fit(x = x_train[:size], \
    y = y_train[:size],\
    batch_size = batch_size,\
    epochs=20,\
    verbose=1,\
    validation_split=0.75)


# %%
MODEL_PATH = '../java/MavenProject/src/main/java/model/'
model.save(MODEL_PATH + 'hit_100_v6.h5')  # save everything in HDF5 format
model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open(MODEL_PATH + "hit_100_v6_config.json", "w") as f:
    f.write(model_json)
model.save_weights(MODEL_PATH + 'hit_100_v6_weights.h5') # save just the weights.

# %%
# New method of saving history

filename = 'hit_100_v6'

# %%

with open('./history/' + filename + '_history.pkl', 'ab') as file_pi:
    pickle.dump(history.history, file_pi)

# %%

# Reading history

with open('./history/' + filename + '_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)


# %%

# summarize history for accuracy
plt.plot(history['root_mean_squared_error'])
plt.plot(history['val_root_mean_squared_error'])
plt.title('model RMSE loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# %%

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# %%
batch = 100
r = np.random.randint(0,y_val.shape[0] - batch)
y_pred = model.predict(x_val[r:r+batch])
print(y_pred.reshape((-1,)))
print(y_val[r:r+batch].reshape((-1,)))
