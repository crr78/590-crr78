"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW3.0A1: Code ANN Regression Using Keras, Train on the Boston Housing Dataset
10/05/2021
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import boston_housing
from keras import models, layers, initializers, activations, regularizers

# load the dataset
bh = boston_housing.load_data()

# assigning pred and response features of the train and test sets to vars
(x_train, y_train), (x_test, y_test) = bh

# checking the number of rows and columns of the train and test sets
print('Input shape (training):', x_train.shape, '\n')
print('Input shape (testing):', x_test.shape, '\n')

# observing output values in the training data
print('Sample of Median Home Prices [$1000s] (training):', y_train[:5], '\n')


#------------------------- 
# Normalizing
#-------------------------

# combine all input values
x = np.vstack([x_train, x_test])

# combine all output values
# y = np.vstack([y_train.reshape(y_train.shape[0], 1), y_test.reshape(y_test.shape[0], 1)])

# calculate mean and standard deviation 
x_mean = np.mean(x, axis = 0)
x_std = np.std(x, axis = 0)
# y_mean = np.mean(y, axis = 0)
# y_std = np.std(y, axis = 0)

# normalize all input values
x_train = (x_train - x_mean)/x_std
x_test = (x_test - x_mean)/x_std

# normalize all output values
# y_train = (y_train - y_mean)/y_std
# y_test = (y_test - y_mean)/y_std


#-----------------------------------
# Train With Keras              
#-----------------------------------

# HYPERPARAMETERS
nodes = 64
act = 'relu' 
opt = 'rmsprop'
loss_func = 'mse'
metric = 'mae'
num_epochs = 500
size = len(x_train)
# kr = 'l2'

# specify input shape
input_shape = (x_train.shape[1],)

# BUILD & COMPILE MODEL
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(units = nodes, activation = act, 
input_shape = input_shape))
	model.add(layers.Dense(units = nodes, activation = act))
	model.add(layers.Dense(units = nodes, activation = act))
	model.add(layers.Dense(1))
	model.compile(optimizer = opt, loss = loss_func, metrics = [metric])
	return model

# IMPLEMENT K-FOLD CROSS-VALIDATION
k = 4
num_val_samples = len(x_train) // k
all_train_mses = []
all_val_mses   = []
all_train_maes = []
all_val_maes   = []
print('Initiating k-fold cross-validation...', '\n')
for i in range(k):
	print('processing fold #', i+1)

	# gather input and output values from train for the kth fold to form the validation set
	x_val = x_train[i*num_val_samples:(i + 1)*num_val_samples]
	y_val = y_train[i*num_val_samples:(i + 1)*num_val_samples]

	# concatenate the remaining input and output values from train to form the training set
	rest_x_train = np.concatenate([x_train[:i*num_val_samples], 
x_train[(i + 1)*num_val_samples:]], axis = 0)
	rest_y_train = np.concatenate([y_train[:i*num_val_samples], 
y_train[(i + 1)*num_val_samples:]], axis = 0)

	# build and compile a model
	model = build_model()

	# fit the model onto the training set, validate the model with the validation set
	history = model.fit(rest_x_train, 
rest_y_train, 
epochs=num_epochs, 
batch_size=len(rest_x_train), 
validation_data=(x_val, y_val), verbose=0)

	# record mse and mae for train and val sets
	train_mse = history.history['loss']
	val_mse   = history.history['val_loss']
	train_mae = history.history['mae']
	val_mae   = history.history['val_mae']
	all_train_mses.append(train_mse)
	all_val_mses.append(val_mse)
	all_train_maes.append(train_mae)
	all_val_maes.append(val_mae)

# for each epoch, calculate the mean mse and mae score for the train set, repeat for the validation set
mean_train_mses = np.array([np.mean([n[i] for n in all_train_mses]) for i in range(num_epochs)])
mean_train_maes = np.array([np.mean([n[i] for n in all_train_maes]) for i in range(num_epochs)])
mean_val_mses   = np.array([np.mean([n[i] for n in all_val_mses]) for i in range(num_epochs)])
mean_val_maes   = np.array([np.mean([n[i] for n in all_val_maes]) for i in range(num_epochs)])

# record the minimum mse and mae for training and validation and the epochs that each of them occur at
min_train_mse = np.min(mean_train_mses)
min_val_mse = np.min(mean_val_mses)
min_train_mse_epoch = np.argmin(mean_train_mses) + 1
min_val_mse_epoch = np.argmin(mean_val_mses) + 1
min_train_mae = np.min(mean_train_maes)
min_val_mae = np.min(mean_val_maes)
min_train_mae_epoch = np.argmin(mean_train_maes) + 1
min_val_mae_epoch = np.argmin(mean_val_maes) + 1
print('-------Results-------')
print('The minimum training mean squared error is', round(min_train_mse, 2), 'and it occurs at epoch #', min_train_mse_epoch)
print('The minimum validation mean squared error is', round(min_val_mse, 2), 'and it occurs at epoch #', min_val_mse_epoch)
print('The minimum training mean absolute error is', round(min_train_mae, 2), 'and it occurs at epoch #', min_train_mae_epoch)
print('The minimum validation mean absolute error is', round(min_val_mae, 2), 'and it occurs at epoch #', min_val_mae_epoch, '\n')

# TRAINING AND VALIDATION LOSS PLOT
epochs = range(1, len(mean_train_mses) + 1)
plt.plot(epochs, mean_train_mses, label = 'Training loss')
plt.plot(epochs, mean_val_mses, label = 'Validation loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# TRAINING AND VALIDATION MAE PLOT
epochs = range(1, len(mean_train_maes) + 1)
plt.plot(epochs, mean_train_maes, label = 'Training MAE')
plt.plot(epochs, mean_val_maes, label = 'Validation MAE')
plt.title('Training & Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# train the final model and evaluate its performance on the test set
model = build_model()
model.fit(x_train,
y_train,
epochs = min_val_mse_epoch,
batch_size = size,
verbose = 0)

mse_test, mae_test = model.evaluate(x_test, y_test)
print('The mean squared error of the model on the testing set is', round(mse_test, 2))
print('The mean absolute error of the model on the testing set is', round(mae_test, 2))

# make predictions from the train and test data using the model
yp_train = model.predict(x_train) 
yp_test  = model.predict(x_test)

# un-normalize actual and predicted values from both train and test
# x_train  = x_std*x_train + x_mean 
# x_test   = x_std*x_test + x_mean
# y_train  = y_std*y_train + y_mean 
# y_test   = y_std*y_test + y_mean 
# yp_train = y_std*yp_train + y_mean 
# yp_test  = y_std*yp_test + y_mean

# PARITY PLOT
plt.plot(yp_train, yp_train, 'k--', label = 'Perfect Performance')
plt.plot(y_train, yp_train, 'o', label = 'Training')
plt.plot(y_test, yp_test, '*', label = 'Test')
plt.title('Parity Plot of Actual and Predicted Median Home Price')
plt.xlabel('Actual Median Home Price [$1000s]')
plt.ylabel('Predicted Median Home Price [$1000s]')
plt.legend()
plt.show()
plt.clf()	
	

