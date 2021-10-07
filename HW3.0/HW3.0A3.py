"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW3.0A3: Code Multi-Class Classification Using Keras, Train on Newswire Dataset
10/05/2021
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers, initializers, activations, regularizers

# load the dataset
rtrs = reuters.load_data(num_words = 10000)

# assigning pred and response features of the train and test sets to vars
(train_data, train_labels), (test_data, test_labels) = rtrs

# checking the number of rows and columns of the train and test sets
print('Input shape (training):', train_data.shape)
print('Input shape (testing):', test_data.shape, '\n')

# observing what inputs look like in the training data
print('Example of encoding for a single newswire:', train_data[0], '\n')

# observing output values
print('Example of encoding for first 5 topics:', train_labels[:5], '\n')


#---------------------------------------
# Data Preprocessing
#---------------------------------------

print('One-hot-encoding the data...', '\n')

# one-hot-encode the newswires to satisfy tensor input requirement for NNs
def ohe(sequences, dim = 10000):

	# initialize a matrix of zeros with shape (len(sequences), dim)
	res = np.zeros((len(sequences), dim))
	
	# for each index, value pair in the list
	for i, seq in enumerate(sequences):
		
		# for each row, assign a 1 to the column number matching the current value
		res[i, seq] = 1.

	# return the one-hot-encoded matrix
	return res

# one-hot-encode the newswires from the training and testing data
x_train_ohe = ohe(train_data)
x_test_ohe  = ohe(test_data)

# the list of topics is not binary since there are more than two, so one-hot-encode these as well
y_train_ohe = to_categorical(train_labels)
y_test_ohe  = to_categorical(test_labels)

print('Sample of first newswire after one-hot-encoding:', x_train_ohe[0], '\n')

print('Splitting the testing set into validation and testing sets...', '\n')

# partitioning test data into val and test sets
f_val = 0.5
indices = np.random.RandomState(seed = 0).permutation(x_test_ohe.shape[0])  # creating a reproducible shuffle of the indices
CUT = int(f_val*x_test_ohe.shape[0])
val_idx = indices[:CUT]
test_idx = indices[CUT:]
x_train, y_train = x_train_ohe, y_train_ohe
x_val, y_val = x_test_ohe[val_idx, :], y_test_ohe[val_idx, :]
x_test, y_test = x_test_ohe[test_idx, :], y_test_ohe[test_idx, :]
print('-------PARTITION INFO--------')
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape, '\n')


#-----------------------------------
# Train With Keras              
#-----------------------------------

print('Building and compiling model...', '\n')

# HYPERPARAMETERS
nodes = 128
act_in = 'sigmoid'
act_hidden = 'relu'
act_out = 'softmax'
opt = 'adamax'
loss_func = 'categorical_crossentropy'
metric = 'accuracy'
num_epochs = 100
size = 512
# kr = 'l2'

# specify input shape
input_shape = (x_train.shape[1],)

# BUILD & COMPILE MODEL
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(units = nodes, activation = act_in, input_shape = input_shape))
	model.add(layers.Dense(46, activation = act_out))
	model.compile(optimizer = opt, loss = loss_func, metrics = [metric])
	return model

# instantiate a compiled model
model = build_model()

print('Training model...', '\n')

# fit the model onto the training set, validate the model with the validation set
history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = size, validation_data = (x_val, y_val), verbose = 0)

# record categorical cross-entropy and accuracy on the train and val sets
train_cce = history.history['loss']
val_cce = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# record the minimum categorical cross-entropy and the maximum accuracy for both train and val, as well as the epochs that each of the extremes occur at
min_train_cce = np.min(train_cce)
min_val_cce = np.min(val_cce)
min_train_cce_epoch = np.argmin(train_cce) + 1
min_val_cce_epoch = np.argmin(val_cce) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1
print('-------Results-------')
print('The minimum training categorical cross-entropy is', round(min_train_cce, 2), 'and it occurs at epoch #', min_train_cce_epoch)
print('The minimum validation categorical cross-entropy is', round(min_val_cce, 2), 'and it occurs at epoch #', min_val_cce_epoch)
print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch))
print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch), '\n')

# TRAINING AND VALIDATION LOSS PLOT
epochs = range(1, len(train_cce) + 1)
plt.plot(epochs, train_cce, label = 'Training loss')
plt.plot(epochs, val_cce, label = 'Validation loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# TRAINING AND VALIDATION ACCURACY PLOT
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, label = 'Training accuracy')
plt.plot(epochs, val_acc, label = 'Validation accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# before training the final model, combine training and validation data so that we only predict on unseen test data

# combine all input values from train and val
x = np.vstack([x_train, x_val])

# combine all output values from train and val
y = np.vstack([y_train, y_val])

# train the final model and evaluate its performance on the test set
model = build_model()
model.fit(x, y, epochs = min_val_cce_epoch, batch_size = size, verbose = 0)
cce_test, acc_test = model.evaluate(x_test, y_test)
print('The categorical cross-entropy of the model on the testing set is', round(cce_test, 2))
print('The accuracy of the model on the testing set is {}%.'.format(round(acc_test*100, 2)))


