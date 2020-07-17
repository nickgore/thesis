"""
Helper functions
"""
from scipy.stats import t
import numpy as np



def vector_to_sequences(vector, sequence_length):
    """
    Converts vector to 2d matrix.
    """
    matrix = []
    for i in range(0, len(vector)-sequence_length+1):
        matrix.append(vector[i:i+sequence_length])
    return matrix


def data_preprocessing(returns, value_at_risk, sequence_length = 20, days_in_advance = 1, split = 0.9):
    """
    Creating input sequences and output vectors for training and testing LSTM model for Value at Risk prediction
    :param returns: vector of returns for financial series (should be normalized for better prediction
    :param value_at_risk: vector of historical VaR, has to have the same length as vector of returns
    :param days_in_advance: time horizon for prediction
    :param sequence_length: length of sequences used for prediction
    :param split: Proportion of data used for training
    Returns input matrix for training, output vector for training, input matrix for testing,
    output vector for testing
    """
    # transform returns vector to sequences
    data_matrix = np.array(vector_to_sequences(returns, sequence_length))

    # split data into training and test sets
    train_row = int(round(split * data_matrix.shape[0]))
    # the training set
    x_train = data_matrix[:train_row, :]
    y_train = value_at_risk[sequence_length - 1 + days_in_advance:train_row + sequence_length - 1 + days_in_advance]
    # the test set
    x_test = data_matrix[train_row:-days_in_advance, :]
    y_test = value_at_risk[train_row + sequence_length - 1 + days_in_advance:, ]

    # adjust shape of input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test


def historical_VaR(returns, rolling_size, alpha = 0.05):
    """
    Computing 20 days realized Value at Risk.
    """
    realized_volatility = returns.rolling(rolling_size).std()
    realized_mean = returns.rolling(rolling_size).mean()

    # Value at Risk based on realized data with t distribution assumption
    df = t.fit(returns)[0]
    z_t = t.ppf([alpha], df=df)
    return realized_mean[:-1] + realized_volatility[:-1] * z_t


def normalization(vector):
    """
    Normalization of returns
    :param vector: original vector
    :return: normalized vector
    """
    return (vector - vector.mean()) / vector.std()
