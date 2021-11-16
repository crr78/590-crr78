"""
Chevy Robertson (crr78@georgetown.edu)
ANLY 590: Neural Networks & Deep Learning
HW6.0: HW6.3.py
11/16/2021
"""


#SOURCE: MODIFIED FROM https://blog.keras.io/building-autoencoders-in-keras.html


#--------
# IMPORTS
#--------

import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt


#----------------------------------------------
# GET DATA SETS & REMOVE "TRUCKS" FROM CIFAR100
#----------------------------------------------

# GET DATA SETS
from keras.datasets import cifar10 as c10
(X, Y), (test_images, test_labels) = c10.load_data()

from keras.datasets import cifar100 as c100
(X_cf, Y_cf), (test_images_cf, test_labels_cf) = c100.load_data()

# REMOVE "TRUCKS" FROM CIFAR100

# initialize a list of indices to keep
non_trucks_idx = []

# loop through each index of the training labels
for i in range(0, len(Y_cf)):

    # if the category is not equal to "truck," which is labeled as 94
    if Y_cf[i] != 94:

        # append the index to the list
        non_trucks_idx.append(i)

# convert the list to an array
non_trucks_idx = np.array(non_trucks_idx)

# use the index to keep only the non-truck samples
X_cf = X_cf[non_trucks_idx, :, :, :]

# repeat for the testing set

non_trucks_idx = []

for i in range(0, len(test_labels_cf)):
    if test_labels_cf[i] != 94:
        non_trucks_idx.append(i)

non_trucks_idx = np.array(non_trucks_idx)

test_images_cf = test_images_cf[non_trucks_idx, :, :, :]


#----------------
# USER PARAMATERS
#----------------

INJECT_NOISE = False
EPOCHS       = 100
NKEEP        = 2500  # DOWNSIZE DATASET
BATCH_SIZE   = 128
N_channels   = 3
PIX          = 32
METRIC       = ['accuracy']


#----------------------
# NORMALIZE AND RESHAPE
#----------------------

X = X.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.
X_cf = X_cf.astype('float32') / 255.
test_images_cf = test_images_cf.astype('float32') / 255.

# DOWNSIZE TO RUN FASTER AND DEBUG
X = X[0:NKEEP]
test_images = test_images[0:NKEEP]
X_cf = X_cf[0:NKEEP]
test_images_cf = test_images_cf[0:NKEEP]


"""
#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)
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


#-------------------
# BUILD CNN-AE MODEL
#-------------------

input_img = keras.Input(shape=(PIX, PIX, N_channels))

#ENCODER
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#DECODER
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(N_channels,(3,3),activation='sigmoid',padding='same')(x)


#----------------------------
# COMPILE & TRAIN AUTOENCODER
#----------------------------

# COMPILE AUTOENCODER
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=METRIC);
print(autoencoder.summary())

# TRAIN AUTOENCODER
history = autoencoder.fit(X_train, X_train, epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          validation_data=(X_val, X_val))


#---------------
# RECORD RESULTS
#---------------

# record binary cross-entropy and accuracy on the train and val sets
train_bce = history.history['loss']          # training loss
val_bce   = history.history['val_loss']      # validation loss
train_acc = history.history['accuracy']      # training accuracy
val_acc   = history.history['val_accuracy']  # validation accuracy

# record the minimum binary cross-entropy and maximum accuracy, as well as the
# epochs that each of the extremes occur at
min_train_bce = np.min(train_bce)
min_val_bce = np.min(val_bce)
min_train_bce_epoch = np.argmin(train_bce) + 1
min_val_bce_epoch = np.argmin(val_bce) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1

# print results
print('-------Results-------')
print('The minimum training binary cross-entropy is', 
      round(min_train_bce, 2), 'and it occurs at epoch #', min_train_bce_epoch)
print('The minimum validation binary cross-entropy is', 
      round(min_val_bce, 2), 'and it occurs at epoch #', min_val_bce_epoch)
print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch))
print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch), '\n')


#------------------
# VISUALIZE RESULTS
#------------------

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


#----------------------
# EVALUATE ON TEST DATA
#----------------------

# evaluate the performance of the autoencoder using the test set

# compile autoencoder
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=METRIC);
print(autoencoder.summary())

# fit autoencoder on entire training set
autoencoder.fit(X, X, epochs=min_val_bce_epoch, batch_size=BATCH_SIZE)

# evaluate model on train and return binary cross-entropy and accuracy
bce_train, acc_train = autoencoder.evaluate(X, X, batch_size=BATCH_SIZE)

# do the same for the testing set
bce_test, acc_test = autoencoder.evaluate(test_images, test_images,
                                          batch_size=BATCH_SIZE)

# print results
print('The binary cross-entropy of the model on the entire training set is',
      round(bce_train, 2))
print('The accuracy of the model on the entire training set is {}%.'.format(round(acc_train*100, 2)))
print('The binary cross-entropy of the model on the entire testing set is',
      round(bce_test, 2))
print('The accuracy of the model on the entire testing set is {}%.'.format(round(acc_test*100, 2)))


#-----------------------------------------------------
# ANOMALY DETECTION USING CIFAR10 & CIFAR100 DATA SETS
#-----------------------------------------------------

# define an error threshold
t = 4*autoencoder.evaluate(X, X, batch_size=X.shape[0])[0]

# use the model to make predictions to form a set of reconstructed images
X1 = autoencoder.predict(X)

# repeat for mnist-fashion
X1_cf = autoencoder.predict(X_cf)

# initialize counters for cifar10 and cifar100
count = 0
count_cf = 0

# loop through cifar10 and cifar100, count number of anomalies for each

# for each index in the training set
for i in range(0, len(X)):

    # add 1 to cifar10 counter if the bce of orig and reconstructed > threshold
    if -np.mean(X[i, :, :].reshape(1, PIX, PIX, N_channels)*np.log(X1[i, :, :].reshape(1, PIX, PIX, N_channels))+(1-X[i, :, :].reshape(1, PIX, PIX, N_channels))*np.log(1-X1[i, :, :].reshape(1, PIX, PIX, N_channels))) > t:
        count += 1

# repeat for cifar 100
for i in range(0, len(X_cf)):
    if -np.mean(X_cf[i, :, :].reshape(1, PIX, PIX, N_channels)*np.log(X1_cf[i, :, :].reshape(1, PIX, PIX, N_channels))+(1-X_cf[i, :, :].reshape(1, PIX, PIX, N_channels))*np.log(1-X1_cf[i, :, :].reshape(1, PIX, PIX, N_channels))) > t:
        count_cf += 1

# record fraction of anomalies for cifar10 and repeat for cifar100
frac = count/len(X)
frac_cf = count_cf/len(X_cf)

# print results
print('The fraction of anomalies in the cifar10 training set is {}'.format(frac))
print('The fraction of anomalies in the cifar100 training set is {}'.format(frac_cf))


#--------------------------------------------
# VISUALIZE ORIGINAL AND RECONSTRUCTED IMAGES
#--------------------------------------------

# MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(test_images)

# VISUALIZE THE RESULTS
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):

    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(test_images[i].reshape(PIX, PIX, N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX, N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#---------------------
# SAVE MODEL & RESULTS
#---------------------

# save model
autoencoder.save('HW6-3_mod.h5')

# results
line1 = '-------Results-------' + '\n' + '\n'

line2 = '-------Training & Validation Results-------' + '\n'

line3 = 'The minimum training binary cross-entropy was ' +  str(round(min_train_bce, 2)) + ' and it occurred at epoch # ' +  str(min_train_bce_epoch) + '\n'

line4 = 'The minimum validation binary cross-entropy was ' + str(round(min_val_bce, 2)) + ' and it occurred at epoch # ' + str(min_val_bce_epoch) + '\n'

line5 = 'The maximum training accuracy was {}% and it occurred at epoch # {}'.format(round(max_train_acc*100, 2), max_train_acc_epoch) + '\n'

line6 = 'The maximum validation accuracy was {}% and it occurred at epoch # {}'.format(round(max_val_acc*100, 2), max_val_acc_epoch) + '\n' + '\n'

line7 = '-------Results Using Entire Training Set-------' + '\n'

line8 = 'The binary cross-entropy of the model on the entire training set was ' + str(round(bce_train, 2)) + '\n'

line9 = 'The accuracy of the model on the entire training set was {}%'.format(round(acc_train*100, 2)) + '\n' + '\n'

line10 = '-------Testing Results-------' + '\n'

line11 = 'The binary cross-entropy of the model on the testing set was ' + str(round(bce_test, 2)) + '\n'

line12 = 'The accuracy of the model on the testing set was {}%'.format(round(acc_test*100, 2)) + '\n' + '\n'

line13 = '-------Anomaly Detection Results-------' + '\n'

line14 = 'The fraction of anomalies in the cifar10 training set was {}'.format(round(frac, 4)) + '\n'

line15 = 'The fraction of anomalies in the cifar100 training set was {}'.format(round(frac_cf, 4)) + '\n' + '\n'

# write results
results_file = open('HW6-3_log.txt', 'w')
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


