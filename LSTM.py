"""
LSTM model for Value at Risk forecast
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.backend as kb


def lstm_for_var():
    """Build LSTM model for Value at Risk forecast"""
    model = Sequential()
    # first layer
    model.add(LSTM(return_sequences = True, input_shape = (None, 1), units = 50))
    model.add(Dropout(0.2))
    # second layer
    model.add(LSTM(return_sequences = False, units = 100))
    model.add(Dropout(0.2))
    # linear activation
    model.add(Dense(units = 1, activation = 'linear'))
    # compile the model
    model.compile(loss = "mse", optimizer = "adam")
    return model
