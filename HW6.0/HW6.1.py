"""
Chevy Robertson (crr78@georgetown.edu)
ANLY 590: Neural Networks & Deep Learning
HW6.0: HW6.1.py
11/16/2021
"""


#--------
# IMPORTS
#--------

import numpy as np
import matplotlib.pyplot as plt
import random
from keras import models
from keras import layers


#----------------
# HYPERPARAMETERS
#----------------

nodes_in = 32                            # number of input layer nodes
n_bottleneck = 10                        # number of bottleneck layer nodes
act_in = 'linear'                        # activation function for input layer
act_hidden = 'linear'                    # activation function for hidden layer
opt = 'rmsprop'                          # optimizer to use
loss_func = 'mean_squared_error'         # loss function to use
metric = ['accuracy']                    # use accuracy as model metric
num_epochs = 40                          # number of epochs
b_size = 1000                            # batch size


#--------------
# GET DATA SETS
#--------------

from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

from keras.datasets import fashion_mnist as fm
(X_fm, Y_fm), (test_images_fm, test_labels_fm) = fm.load_data()


#----------------------
# NORMALIZE AND RESHAPE
#----------------------

# rescale the images in the training set by the largest pixel value
X = X/np.max(X)

# shape should be consistent with num samples, and h*w of pixels in each image 
X = X.reshape(60000, 28*28)

# repeat for the images in the testing set
test_images = test_images/np.max(test_images)
test_images = test_images.reshape(10000, 28*28)

# repeat for the images in the fm training set
X_fm = X_fm/np.max(X_fm)
X_fm = X_fm.reshape(60000, 28*28)

# repeat for the images in the fm testing set
test_images_fm = test_images_fm/np.max(test_images_fm)
test_images_fm = test_images_fm.reshape(10000, 28*28)


"""
# as an option, use subsets of the data for debugging
X = X[0:int(len(X)/100)]
test_images  = test_images[0:int(len(test_images)/100)]
X_fm = X_fm[0:int(len(X_fm)/100)]
test_images_fm  = test_images_fm[0:int(len(test_images_fm)/100)]
"""


#-------------------------------------
# PARTITION INTO TRAINING & VALIDATION
#-------------------------------------

# specify percentage of training data set to use for training subset
f_train = 0.8

# make a reproducible shuffle of the indices of the training set
indices = np.random.RandomState(seed=0).permutation(len(X))

# specify the index for splitting the training set into training and validation
CUT = int(f_train*len(X))

# use the first 80% of the random indices to form the training data set
train_idx = indices[:CUT]

# use the last 20% for the validation set
val_idx = indices[CUT:]

# use the random indices to form the training and validation sets
X_train = X[train_idx, :]
X_val   = X[val_idx, :]


#------------
# BUILD MODEL
#------------

# using a simple 3-layer basic AE including bottleneck layer
model = models.Sequential()
model.add(layers.Dense(nodes_in, activation=act_in, input_shape=(28*28, )))
model.add(layers.Dense(n_bottleneck, activation=act_hidden))
model.add(layers.Dense(28*28,  activation='linear'))
print(model.summary())


#--------------------
# COMPILE & FIT MODEL
#--------------------

# compile model, specifying optimizer, loss, and evaluation metrics to us
model.compile(optimizer=opt, loss=loss_func, metrics=metric)

# fit the model using the train set, validate the model using the validation set
history = model.fit(X_train, X_train, epochs=num_epochs, batch_size=b_size,
                    validation_data=(X_val, X_val))


#---------------
# RECORD RESULTS
#---------------

# record mean squared error and accuracy on the train and val sets
train_mse = history.history['loss']          # training loss
val_mse   = history.history['val_loss']      # validation loss
train_acc = history.history['accuracy']      # training accuracy
val_acc   = history.history['val_accuracy']  # validation accuracy

# record the minimum mean squared error and maximum accuracy, as well as the
# epochs that each of the extremes occur at
min_train_mse = np.min(train_mse)
min_val_mse = np.min(val_mse)
min_train_mse_epoch = np.argmin(train_mse) + 1
min_val_mse_epoch = np.argmin(val_mse) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1

# print results
print('-------Results-------')
print('The minimum training mean squared error is', 
      round(min_train_mse, 2), 'and it occurs at epoch #', min_train_mse_epoch)
print('The minimum validation mean squared error is', 
      round(min_val_mse, 2), 'and it occurs at epoch #', min_val_mse_epoch)
print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch))
print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch), '\n')


#------------------
# VISUALIZE RESULTS
#------------------

# TRAINING AND VALIDATION LOSS PLOT
epochs = range(1, len(train_mse) + 1)
plt.plot(epochs, train_mse, label = 'Training loss')
plt.plot(epochs, val_mse, label = 'Validation loss')
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


#----------------------
# EVALUATE ON TEST DATA
#----------------------

# evaluate the model's performance using the test set

# build a simple 3-layer basic AE including bottleneck layer
model = models.Sequential()
model.add(layers.Dense(nodes_in, activation=act_in, input_shape=(28*28, )))
model.add(layers.Dense(n_bottleneck, activation=act_hidden))
model.add(layers.Dense(28*28,  activation='linear'))
print(model.summary())

# compile model
model.compile(optimizer=opt, loss=loss_func, metrics=metric)

# fit model on entire training set
model.fit(X, X, epochs=min_val_mse_epoch, batch_size=b_size)

# evaluate model on train and return mean squared error and accuracy
mse_train, acc_train = model.evaluate(X, X, batch_size=b_size)

# do the same for the testing set
mse_test, acc_test = model.evaluate(test_images, test_images, batch_size=b_size)

# print results
print('The mean squared error of the model on the entire training set is',
      round(mse_train, 2))
print('The accuracy of the model on the entire training set is {}%.'.format(round(acc_train*100, 2)))
print('The mean squared error of the model on the entire testing set is',
      round(mse_test, 2))
print('The accuracy of the model on the entire testing set is {}%.'.format(round(acc_test*100, 2)))


#--------------------------------------------------------
# ANOMALY DETECTION USING MNIST & MNIST-FASHION DATA SETS
#--------------------------------------------------------

# define an error threshold
t = 4*model.evaluate(X, X, batch_size=X.shape[0])[0]

# use the model to make predictions to form a set of reconstructed images
X1 = model.predict(X)

# repeat for mnist-fashion
X1_fm = model.predict(X_fm)

# initialize counters for mnist and mnist-fashion
count_m  = 0
count_fm = 0

# loop through mnist and mnist-fashion, count number of anomalies for each

# for each index in the training set
for i in range(0, len(X)):

    # add 1 to mnist counter if the mse of orig and reconstructed > threshold
    if np.mean((X[i,:].reshape(1,28*28)-X1[i,:].reshape(1,28*28))**2) > t:
        count_m += 1

    # do the same for each image in fashion mnist
    if np.mean((X_fm[i,:].reshape(1,28*28)-X1_fm[i,:].reshape(1,28*28))**2) > t:
        count_fm += 1

# record fraction of anomalies for mnist and repeat for fashion-mnist
frac_m  = count_m/len(X)
frac_fm = count_fm/len(X)

# print results
print('The fraction of anomalies in the mnist training set is {}'.format(frac_m))
print('The fraction of anomalies in the fashion mnist training set is {}'.format(frac_fm))


#--------------------------------------------
# VISUALIZE ORIGINAL AND RECONSTRUCTED IMAGES
#-------------------------------------------- 

# reshape the images in order to visualize them
X  = X.reshape(X.shape[0], 28, 28)
X1 = X1.reshape(X1.shape[0], 28, 28)

# compare random original images to corresponding reconstructed version

# plot the images as 4 rows and 1 column
f, ax = plt.subplots(4, 1)

# choose random indices to use for selecting images from the data set
img_idx1 = random.randint(0, len(X)-1)
img_idx2 = random.randint(0, len(X)-1)

# plot the original images and their reconstructions
ax[0].imshow(X[img_idx1])
ax[1].imshow(X1[img_idx1])
ax[2].imshow(X[img_idx2])
ax[3].imshow(X1[img_idx2])
plt.show()


#---------------------
# SAVE MODEL & RESULTS
#---------------------

# save model
model.save('HW6-1_mod.h5')

# results
line1 = '-------Results-------' + '\n' + '\n'

line2 = '-------Training & Validation Results-------' + '\n'

line3 = 'The minimum training mean squared error was ' +  str(round(min_train_mse, 2)) + ' and it occurred at epoch # ' +  str(min_train_mse_epoch) + '\n'

line4 = 'The minimum validation mean squared error was ' + str(round(min_val_mse, 2)) + ' and it occurred at epoch # ' + str(min_val_mse_epoch) + '\n'

line5 = 'The maximum training accuracy was {}% and it occurred at epoch # {}'.format(round(max_train_acc*100, 2), max_train_acc_epoch) + '\n'

line6 = 'The maximum validation accuracy was {}% and it occurred at epoch # {}'.format(round(max_val_acc*100, 2), max_val_acc_epoch) + '\n' + '\n'

line7 = '-------Results Using Entire Training Set-------' + '\n'

line8 = 'The mean squared error of the model on the entire training set was ' + str(round(mse_train, 2)) + '\n'

line9 = 'The accuracy of the model on the entire training set was {}%'.format(round(acc_train*100, 2)) + '\n' + '\n'

line10 = '-------Testing Results-------' + '\n'

line11 = 'The mean squared error of the model on the testing set was ' + str(round(mse_test, 2)) + '\n'

line12 = 'The accuracy of the model on the testing set was {}%'.format(round(acc_test*100, 2)) + '\n' + '\n'

line13 = '-------Anomaly Detection Results-------' + '\n'

line14 = 'The fraction of anomalies in the mnist training set was {}'.format(round(frac_m, 4)) + '\n'

line15 = 'The fraction of anomalies in the fashion mnist training set was {}'.format(round(frac_fm, 4)) + '\n' + '\n'

# write results
results_file = open('HW6-1_log.txt', 'w')
results_file.write(line1)
results_file.write(line2)
results_file.write(line3)
results_file.write(line4)
results_file.write(line5)
results_file.write(line6)
results_file.write(line7)
results_file.write(line8)
results_file.write(line9)
results_file.write(line10)
results_file.write(line11)
results_file.write(line12)
results_file.write(line13)
results_file.write(line14)
results_file.write(line15)
results_file.close()


