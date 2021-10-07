"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW3.0A2: Code Binary Classification Using Keras, Train on IMDB Dataset
10/05/2021
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras import models, layers, initializers, activations, regularizers
from sklearn.metrics import confusion_matrix

# load the dataset
imdb = imdb.load_data(num_words = 3000)

# assigning pred and response features of the train and test sets to vars
(train_data, train_labels), (test_data, test_labels) = imdb

# checking the number of rows and columns of the train and test sets
print('Input shape (training):', train_data.shape)
print('Input shape (testing):', test_data.shape, '\n')

# observing what inputs look like in the training data
print('Example of encoding for a single review:', train_data[0], '\n')

# observing output values
print('Example of encoding for 5 ratings (0 indicates negative, 1 indicate positive):', train_labels[:5], '\n')


#---------------------------------------
# Data Preprocessing
#---------------------------------------

print('One-hot-encoding the data...', '\n')

# one-hot-encode the reviews to satisfy tensor input requirement for NNs
def ohe(sequences, dim = 3000):

	# initialize a matrix of zeros with shape (len(sequences), dim)
	res = np.zeros((len(sequences), dim))
	
	# for each index, value pair in the list
	for i, seq in enumerate(sequences):
		
		# for each row, assign a 1 to the column number matching the current value
		res[i, seq] = 1.

	# return the one-hot-encoded matrix
	return res

# one-hot-encode the reviews from the training and testing data
x_train_ohe = ohe(train_data)
x_test_ohe  = ohe(test_data)

# the list of ratings are already binary, so just vectorize them and convert the ints to floats
y_train_ohe = np.asarray(train_labels).astype('float32')
y_test_ohe  = np.asarray(test_labels).astype('float32')

print('Sample of first review after one-hot-encoding:', x_train_ohe[0], '\n')

print('Splitting the testing set into validation and testing sets...', '\n')

# partitioning test data into val and test sets
f_val = 0.5
indices = np.random.RandomState(seed = 0).permutation(x_test_ohe.shape[0])  # creating a reproducible shuffle of the indices
CUT = int(f_val*x_test_ohe.shape[0])
val_idx = indices[:CUT]
test_idx = indices[CUT:]
x_train, y_train = x_train_ohe, y_train_ohe
x_val, y_val = x_test_ohe[val_idx, :], y_test_ohe[val_idx]
x_test, y_test = x_test_ohe[test_idx, :], y_test_ohe[test_idx]
print('-------PARTITION INFO--------')
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape, '\n')


#-----------------------------------
# Train With Keras              
#-----------------------------------

print('Building and compiling model...', '\n')

# HYPERPARAMETERS
nodes = 16
act_in = 'sigmoid'
act_hidden = 'relu'
act_out = 'sigmoid'
opt = 'rmsprop'
loss_func = 'binary_crossentropy'
metric = 'accuracy'
num_epochs = 20
size = 256
# kr = 'l2'

# specify input shape
input_shape = (x_train.shape[1],)

# BUILD & COMPILE MODEL
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(units = nodes, activation = act_in, input_shape = input_shape))
	model.add(layers.Dense(units = nodes, activation = act_hidden))
	model.add(layers.Dense(1, activation = act_out))
	model.compile(optimizer = opt, loss = loss_func, metrics = [metric])
	return model

# instantiate a compiled model
model = build_model()

print('Training model...', '\n')

# fit the model onto the training set, validate the model with the validation set
history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = size, validation_data = (x_val, y_val), verbose = 0)

# record binary cross-entropy and accuracy on the train and val sets
train_bce = history.history['loss']
val_bce = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# record the minimum binary cross-entropy and the maximum accuracy for both train and val, as well as the epochs that each of the extremes occur at
min_train_bce = np.min(train_bce)
min_val_bce = np.min(val_bce)
min_train_bce_epoch = np.argmin(train_bce) + 1
min_val_bce_epoch = np.argmin(val_bce) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1
print('-------Results-------')
print('The minimum training binary cross-entropy is', round(min_train_bce, 2), 'and it occurs at epoch #', min_train_bce_epoch)
print('The minimum validation binary cross-entropy is', round(min_val_bce, 2), 'and it occurs at epoch #', min_val_bce_epoch)
print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch))
print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch), '\n')

# TRAINING AND VALIDATION LOSS PLOT
epochs = range(1, len(train_bce) + 1)
plt.plot(epochs, train_bce, label = 'Training loss')
plt.plot(epochs, val_bce, label = 'Validation loss')
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
y = np.vstack([y_train.reshape(y_train.shape[0], 1), y_val.reshape(y_val.shape[0], 1)])

# train the final model and evaluate its performance on the test set
model = build_model()
model.fit(x, y, epochs = min_val_bce_epoch, batch_size = size, verbose = 0)
bce_test, acc_test = model.evaluate(x_test, y_test)
print('The binary cross-entropy of the model on the testing set is', round(bce_test, 2))
print('The accuracy of the model on the testing set is {}%.'.format(round(acc_test*100, 2)))

# make predictions from the train, val, and test data using the model
yp_train = model.predict(x_train)
yp_val   = model.predict(x_val) 
yp_test  = model.predict(x_test)

# round the predicted probabilities to the nearest digit to indicate either 0 (negative prediction) or 1 (positive prediction)
yp_train = np.around(yp_train)
yp_val   = np.around(yp_val)
yp_test  = np.around(yp_test)


#----------------# 
# Results
#----------------#

# TRAINING PREDICTIONS 
train_TP = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[1][1]*100
train_TN = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[0][0]*100
train_FP = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[0][1]*100
train_FN = (confusion_matrix(y_train, yp_train)/y_train.shape[0])[1][0]*100

# form a list of the categories for possible prediction types
cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']

# store the percentage of each prediction type
pcts = [train_TP, train_TN, train_FP, train_FN]

# PLOT TRAINING PREDICTIONS
plt.bar(cats, pcts)
plt.title('Model Results (Training Set)')
plt.xlabel('Prediction Type')
plt.ylabel('Percentage')
plt.show()

# VALIDATION PREDICTIONS 
val_TP = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[1][1]*100
val_TN = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[0][0]*100
val_FP = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[0][1]*100
val_FN = (confusion_matrix(y_val, yp_val)/y_val.shape[0])[1][0]*100

# form a list of the categories for possible prediction types
cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']

# store the percentage of each prediction type
pcts = [val_TP, val_TN, val_FP, val_FN]

# PLOT VALIDATION PREDICTIONS
plt.bar(cats, pcts)
plt.title('Model Results (Validation Set)')
plt.xlabel('Prediction Type')
plt.ylabel('Percentage')
plt.show()

# TESTING PREDICTIONS 
test_TP = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[1][1]*100
test_TN = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[0][0]*100
test_FP = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[0][1]*100
test_FN = (confusion_matrix(y_test, yp_test)/y_test.shape[0])[1][0]*100

# form a list of the categories for possible prediction types
cats = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']

# store the percentage of each prediction type
pcts = [test_TP, test_TN, test_FP, test_FN]

# PLOT TESTING PREDICTIONS
plt.bar(cats, pcts)
plt.title('Model Results (Testing Set)')
plt.xlabel('Prediction Type')
plt.ylabel('Percentage')
plt.show()


