# %%



# Import statement

import json

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

## Keras
import keras
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Concatenate
from keras.losses import categorical_crossentropy
from keras.activations import sigmoid, softmax, relu, tanh
from keras.metrics import Accuracy
from keras.optimizers import Adam

import pickle





def import_csv(filename: str, separate_rate: float=0.75):
    data = pd.read_csv(filename)

    # Comment to also get the unpicking data
    data = data.loc[data['Label'] == 1]

    x_df = data.iloc[:, :-1]
    y_df = data.iloc[:, -1:]

    separate_rate = separate_rate
    n_training = int(separate_rate * x_df.shape[0])

    x_train_hand = x_df.iloc[:n_training,:52].values
    x_train_face = x_df.iloc[:n_training,52:].values
    y_train = y_df.iloc[:n_training,:].values
    x_val_hand = x_df.iloc[n_training:,:52].values
    x_val_face =x_df.iloc[n_training:,52:].values
    y_val = y_df.iloc[n_training:,:].values
    n_match = data.shape[0]
    n_match_train = y_train.shape[0]
    n_match_val = y_val.shape[0]

    return (x_train_hand, x_train_face, y_train, n_match_train), (x_val_hand, x_val_face, y_val, n_match_val), n_match

def build_model():
    '''
    input: hand and face
    output: pick or not
    '''
    in_hand = Input(shape=(52,))
    in_face = Input(shape=(52,))

    tensor_hand = Dense(128, activation='relu', kernel_initializer='random_normal') (in_hand)

    tensor_face = Dense(128, activation='relu', kernel_initializer='random_normal') (in_face)

    tensor = Concatenate(axis=1) ([tensor_hand, tensor_face])

    tensor = Dense(64, activation='relu', kernel_initializer='random_normal') (tensor)

    output = Dense(1, activation='sigmoid', kernel_initializer='random_normal')(tensor)
    
    model = Model([in_hand, in_face], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()
    return model

def build_model_2():
    '''
    input: face
    output: hand
    '''
    in_face = Input(shape=(52,))

    # tensor_hand = Dense(128, activation='relu', kernel_initializer='random_normal') (in_hand)

    tensor_face = Dense(128, activation='relu', kernel_initializer='random_normal') (in_face)

    tensor = Dense(64, activation='relu', kernel_initializer='random_normal') (tensor_face)

    output = Dense(52, activation='sigmoid', kernel_initializer='random_normal')(tensor)
    
    model = Model(in_face, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()
    return model

# %%

(hand_train, face_train, y_train, n_train), (hand_val, face_val, y_val, n_val), n_match = \
    import_csv("./GinRummyMavenProject/data_picking.csv", 0.75)


hand_face = np.multiply(hand_train, face_train)

# %%

model = build_model_2()


# %%
size = y_train.shape[0]
batch_size = 1000
model.fit(x = face_train[:size], \
    y = hand_face[:size],\
    batch_size = batch_size,\
    epochs=100,\
    verbose=1,\
    validation_split=0.75)


# %%

model.save('drawing.h5')  # save everything in HDF5 format
model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open("drawing_config.json", "w") as f:
    f.write(model_json)
model.save_weights('drawing_weights.h5') # save just the weights.


# %%
r = np.random.randint(0, y_val.shape[0])
y_pred = model.predict(face_val[r:r+1])
print(y_pred)

print(np. multiply(hand_val[r:r+1], face_val[r:r+1]))

# %%

face_val[r:r+1]

# %%

hand = np.zeros((1,52))
i = 10
while i > 0:
    r = np.random.randint(0, 52)
    if hand[0][r] != 0:
        continue
    hand[0][r] = 1/42
    i-=1

# %%


hand
# %%
face = np.zeros((1, 52))
face[0][18] = 1

y_pred = model.predict([hand, face])
y_pred 
# %%

hand

#%%



def 




























# %%

#Hand Eval
filename = "./GinRummyMavenProject/data_linear.csv"

data = pd.read_csv(filename)

x_train = data.iloc[:, 0:52].values
y_train = data.iloc[:, 52:].values 

# %%
import json

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

## Keras
import keras
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Concatenate
from keras.losses import categorical_crossentropy
from keras.activations import sigmoid, softmax, relu, tanh
from keras.metrics import Accuracy
from keras.optimizers import Adam

def build_model():
    input = Input(shape=(52,))
    tensor = Dense(128, activation='relu', kernel_initializer='random_normal') (input)

    tensor = Dense(56, activation='relu', kernel_initializer='random_normal') (tensor)
    out = Dense(1, activation='sigmoid', kernel_initializer='random_normal') (tensor)

    model = Model(input, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

# %%

x_train.shape

# %%%%%%%%%%%%%

model = build_model()

# %%
size = x_train.shape[0]
batch_size = 100
model.fit(x = hand_train[:size], \
    y = y_train[:size],\
    batch_size = batch_size,\
    epochs=1,\
    verbose=1,\
    validation_split=0.75)
