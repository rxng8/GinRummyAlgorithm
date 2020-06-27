# %%

import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

# %%

# instantiate flask 
app = flask.Flask(__name__)

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
# global graph
# graph = tf.get_default_graph()
# model = load_model('./simple_training.h5', custom_objects={'auc': auc})
model = load_model('./simple_training.h5')

# define a predict function as an endpoint 
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        raise KeyError()

    # if parameters are found, return a prediction
    if (params != None):
        return params

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')