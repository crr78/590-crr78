"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW3.0A1: Code ANN Regression Using Keras, Train on the Boston Housing Dataset
10/05/2021
"""

# imports
import numpy as np
from tensorflow import keras
from keras.datasets import boston_housing
from keras import models, layers, initializers, activations

# load the dataset
bh = boston_housing.load_data()

# assigning pred and response features of the train and test sets to vars
(train_data, train_targets), (test_data, test_targets) = bh

# checking the number of rows and columns of the train and test sets
print(train_data.shape)
print(test_data.shape)

# observing output values in the training data
print(train_targets[:5])

# normalizing the values within each feature
train_mean = train_data.mean(axis = 0)
train_std = train_data.std(axis = 0)
test_mean = test_data.mean(axis = 0)
test_std = test_data.std(axis = 0)
x_train = (train_data - train_mean)/train_std
y_train = (test_data - test_mean)/test_std

#-------------------------
# Build Model             
#-------------------------

# hyperparameters
optimizer = 'rmsprop'
loss_function = 'MeanSquaredError' 
# loss_function="MeanAbsoluteError" 
learning_rate = 0.051
numbers_epochs = 200
model_type = 'linear'
input_shape = (x_train.shape[1],)

"""
# assign predictor and response features of the train and test sets to vars
(x_train, y_train), (test_data, test_targets) = bh

# checking the number of rows and cols of the train and test sets
print(x_train.shape)
print(test_data.shape)

# observing output values in the training data
print(y_train[:5])

# partitioning test data into val and test sets
f_val  = 0.75
f_test = 0.25
indices = np.random.permutation(test_data.shape[0])
CUT = int(f_val*test_data.shape[0])
val_idx = indices[:CUT]
test_idx = indices[CUT:]
x_val, y_val = test_data[val_idx, :], test_targets[val_idx]
x_test, y_test = test_data[test_idx, :], test_targets[test_idx]
print('------PARTITION INFO---------')
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# combining input and outputs from train and test to determine mean and std prior to normalizing

# join x data from train and test by row
x = np.vstack([x_train, test_data])

# join y data from train and test by row
y = np.vstack([y_train.reshape(y_train.shape[0], 1), test_targets.reshape(test_targets.shape[0], 1)])

# normalizing
x_mean  = np.mean(x, axis = 0)
x_std   = np.std(x, axis = 0)
y_mean  = np.mean(y, axis = 0)
y_std   = np.std(y, axis = 0)
x_train = (x_train - x_mean)/x_std
x_val   = (x_val - x_mean)/x_std
x_test  = (x_test - x_mean)/x_std
y_train = (y_train - y_mean)/y_std
y_val   = (y_val - y_mean)/y_std
y_test  = (y_test - y_mean)/y_std
"""





