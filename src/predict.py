import pandas as pd 
import numpy as np 
 
from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model
from src.train_pipeline import layer_neurons_weighted_sum, layer_neurons_output

h = [None]*config.NUM_LAYERS
z = [None]*config.NUM_LAYERS

def inference(X,trained_biases,trained_weights):
    h[0] = X.reshape(1,X.shape[0])

    for l in range(1,config.NUM_LAYERS):
        z[l] = layer_neurons_weighted_sum(h[l-1], trained_biases, trained_weights)

        h[l] = layer_neurons_output(z[l], config.f[l])


        return int(h[l] > 0.5)