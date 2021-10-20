"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW4.0: MNIST, MNIST Fashion, CIFAR-10
10/20/2021
"""

#--------
# IMPORTS
#--------

from keras import layers 
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


#----------------
# CODE PARAMETERS
#----------------

# specify which dataset to use
dataset = 'mnist'
# dataset = 'fashion_mnist'
# dataset = 'cifar10'

# specify dims of each image based on dataset being used

if dataset == 'mnist' or dataset == 'fashion_mnist':
    h = 28  # height in pixels of each image
    w = 28  # width in pixels of each image
    c = 1   # number of color channels for each image

else:
    h = 32
    w = 32
    c = 3

# flag for implementing data augmentation
data_aug = False


#----------------
# HYPERPARAMETERS
#----------------

nodes_in = 32                            # number of input layer nodes
nodes_hidden = 32                        # number of hidden layer nodes
nodes_out = 10                           # number of output layer nodes
ks = (3, 3)                              # kernel size
ps = (2, 2)                              # pool size
act_in = 'relu'                          # activation function for input layer
act_hidden = 'relu'                      # activation function for hidden layer
act_out = 'softmax'                      # activation function for output layer
opt = 'rmsprop'                          # optimizer to use
loss_func = 'categorical_crossentropy'   # loss function to use
metric = ['accuracy']                    # use accuracy as model metric
num_epochs = 40                          # number of epochs
b_size = 32                              # batch size
model_type = 'cnn'                       # use CNN model
# model_type = 'dff'                     # use DFF model


#----------------------------------------
# BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#----------------------------------------

# function for building DFF model as benchmark
def build_dff():

    # create an instance of a Keras Sequential model
    model = models.Sequential()

    # specify input layer, reshape input in input_shape argument
    model.add(layers.Dense(nodes_in, activation=act_in, input_shape=(h*w*c,)))

    # hidden layers
    model.add(layers.Dense(nodes_hidden, activation=act_hidden))
    model.add(layers.Dense(nodes_hidden, activation=act_hidden))

    # output layer, softmax activation function
    model.add(layers.Dense(nodes_out, activation=act_out))

    # prints summary of model
    model.summary()

    # return the model
    return model

# function for building CNN
def build_cnn():

    # create an instance of a Keras Sequential model
    model = models.Sequential()

    # incorporate an instance of a convolutional layer
    model.add(layers.Conv2D(nodes_in, ks, activation=act_in,
                            input_shape=(h, w, c)))

    # incorporate an instance of a MaxPooling layer
    model.add(layers.MaxPooling2D(ps))

    model.add(layers.Conv2D(nodes_hidden, ks, activation=act_hidden)) 
    model.add(layers.MaxPooling2D(ps))
    model.add(layers.Conv2D(nodes_hidden, ks, activation=act_hidden))

    # reshape the array to for correct input to the DFF
    model.add(layers.Flatten())
    model.add(layers.Dense(nodes_hidden, activation=act_hidden))
    model.add(layers.Dense(nodes_out, activation=act_out))
    model.summary()
    return model

# build dff or cnn based on choice of model type
if model_type == 'dff':
    model = build_dff()
else:
    model = build_cnn()


#----------------------
# GET DATA AND REFORMAT
#----------------------

# loading mnist and assigning images and labels for both train and test
if dataset == 'mnist':
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# loading fashion mnist and assigning images and labels for both train and test
elif dataset == 'fashion_mnist':
    from keras.datasets import fashion_mnist as fm
    (train_images, train_labels), (test_images, test_labels) = fm.load_data()

# loading cifar-10 and assigning images and labels for both train and test
else:
    from keras.datasets import cifar10 as c10
    (train_images, train_labels), (test_images, test_labels) = c10.load_data()


#-----------------------
# VISUALIZE RANDOM IMAGE 
#-----------------------

# function to visualize a random image in the data set
def viz_rand_img():

    # combine all images row-wise
    all_images = np.vstack([train_images, test_images])

    # choose a random index to use for selecting an image from the data set
    img_idx = random.randint(0, len(all_images)-1)

    # plot the random image
    plt.imshow(all_images[img_idx])

    # title the plot
    plt.title('A Random Image From the Training Set')

    # show the plot
    plt.show()

# function to call for visualizing random image
viz_rand_img()


# reformat shape of train and test based on model type
if model_type == 'dff':
    train_images = train_images.reshape((train_images.shape[0], h*w*c))
    test_images  = test_images.reshape((test_images.shape[0], h*w*c))
else:
    train_images = train_images.reshape((train_images.shape[0], h, w, c))
    test_images  = test_images.reshape((test_images.shape[0], h, w, c))

# as an option, use subset of the data for debugging
train_images = train_images[0:int(len(train_images)/100)]
test_images  = test_images[0:int(len(test_images)/100)]
train_labels = train_labels[0:int(len(train_labels)/100)]
test_labels  = test_labels[0:int(len(test_labels)/100)]


#----------
# NORMALIZE
#----------

# unless we opt to do data augmentation
if (not data_aug):

    # rescale the the images in both the training and testing sets
    train_images = train_images.astype('float32') / 255 
    test_images  = test_images.astype('float32') / 255

# if we do, perform the normalizaiton during data augmentation
else:
    print('Normalization will occur during data augmentation.')


#-----------------------------
# PARTITIONING (TRAIN AND VAL)
#-----------------------------

# specify percentage of training data set to use for training subset
f_train = 0.8

# make a reproducible shuffle of the indices of the training set
indices = np.random.RandomState(seed=0).permutation(len(train_images))

# specify the index for splitting the training set into training and validation
CUT = int(f_train*len(train_images))

# use the first 80% of the random indices to form the training data set
train_idx = indices[:CUT]

# use the last 20% for the validation set
val_idx = indices[CUT:]

# using the random indices to form the training and validation sets

# make sure to use a different shape for the DFF model
if model_type == 'dff':
    x_train = train_images[train_idx, :]
    x_val   = train_images[val_idx, :]

# do the same for the CNN model
else:
    x_train = train_images[train_idx, :, :]
    x_val   = train_images[val_idx, :, :]

# grab the train and val labels that correspond to the train and val images
y_train = train_labels[train_idx]
y_val   = train_labels[val_idx]


#-------------------------------------------------------
# CONVERT CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX
#-------------------------------------------------------

# show what the label looks like before conversion
tmp = y_train[0]

# one-hot-encode the training and validation labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# one-hot-encode the actual training set and test set as well
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# showing conversion of label to one-hot-encoded version
print(tmp, '-->', y_train[0])
print("train_labels shape:", y_train.shape)


#----------------------------------------------------------------------
# COMPILE AND TRAIN MODEL (AND POTENTIALLY IMPLEMENT DATA AUGMENTATION)
#----------------------------------------------------------------------

# compile model, specifying optimizer, loss, and evaluation metric to use
model.compile(optimizer=opt, loss=loss_func, metrics=metric)

# implementing data augmentation

# if the flag specifies to augment the data
if data_aug == True:

    # set up a data augmentation configuration for train via ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,         # normalizes 
                                       rotation_range=40,        # rotation
                                       width_shift_range=0.2,    # shifts horiz
                                       height_shift_range=0.2,   # shifts vert
                                       shear_range=0.2,          # shearing
                                       zoom_range=0.2,           # zooming
                                       horizontal_flip=True)     # flips horiz

    # do not augment the validation data, only normalize it
    test_datagen = ImageDataGenerator(rescale=1./255)

    # instantate the generator to be used to apply the augmentations to train
    train_generator = train_datagen.flow(x_train, y_train, batch_size=1)

    # do the same for the validation data
    validation_generator = test_datagen.flow(x_val, y_val, batch_size=1)

    # fit model with fit_generator instead of fit since we are using generators 
    history = model.fit_generator(train_generator, steps_per_epoch=20,
                                  epochs=20,
                                  validation_data=validation_generator,
                                  validation_steps=10, verbose=0)
else:
    # if we are not doing data augmentation, just fit the model like usual
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=b_size,
                        validation_data = (x_val, y_val), verbose=0)

# record categorical cross-entropy and accuracy on the train and val sets
train_cce = history.history['loss']         # training loss
val_cce = history.history['val_loss']       # validation loss
train_acc = history.history['accuracy']     # training accuracy
val_acc = history.history['val_accuracy']   # validation accuracy

# record the minimum categorical cross-entropy and the maximum accuracy for 
# both train and val, as well as the epochs that each of the extremes occur at
min_train_cce = np.min(train_cce)
min_val_cce = np.min(val_cce)
min_train_cce_epoch = np.argmin(train_cce) + 1
min_val_cce_epoch = np.argmin(val_cce) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1

# print results
print('-------Results-------')
print('The minimum training categorical cross-entropy is', 
      round(min_train_cce, 2), 'and it occurs at epoch #', min_train_cce_epoch)
print('The minimum validation categorical cross-entropy is', 
      round(min_val_cce, 2), 'and it occurs at epoch #', min_val_cce_epoch)
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

#----------------------
# EVALUATE ON TEST DATA
#----------------------

# train the final model and evaluate its performance on the test set

# choose model to use conditioned on "model_type" parameter
if model_type == 'dff':
    model = build_dff()
else:
    model = build_cnn()

# compile model
model.compile(optimizer=opt, loss=loss_func, metrics=metric)

# fit model on entire training set
model.fit(train_images, train_labels, epochs=min_val_cce_epoch,
          batch_size=b_size, verbose=0)

# evaluate model on train and return categorical cross-entropy and accuracy
cce_train, acc_train = model.evaluate(train_images, train_labels,
                                      batch_size=b_size)

# do the same for the testing set
cce_test, acc_test = model.evaluate(test_images, test_labels, batch_size=b_size)

# print results
print('The categorical cross-entropy of the model on the training set is',
      round(cce_train, 2))
print('The accuracy of the model on the training set is {}%.'.format(round(acc_train*100, 2)))
print('The categorical cross-entropy of the model on the testing set is',
      round(cce_test, 2))
print('The accuracy of the model on the testing set is {}%.'.format(round(acc_test*100, 2)))
        

# methods included for saving a model, hyperparameters, and reading model

if dataset == 'mnist':
    model.save('mnist_mod.h5')  # save model and hyperparameters
    model = models.load_model('mnist_mod.h5')  # load model and hyperparameters
elif dataset == 'fashion_mnist':
    model.save('fashion_mnist_mod.h5')
    model = models.load_model('fashion_mnist_mod.h5')
else:
    model.save('cifar10_mod.h5')
    model = models.load_model('cifar10_mod.h5')

# print summary of loaded model
print(model.summary())

#-----------------------------------
# VISUALIZE INTERMEDIATE ACTIVATIONS
#-----------------------------------

# viz an intermediate activation according to image, layer, and channel number
def viz_acts(img_num, lyr_num, c_num):

    # only makes sense to visualize the activations from conv or pooling layers
    if model_type == 'dff':
        print('Function only plots activations from Conv or MaxPooling layers')

    # for models built with the conv or pooling layers
    else:

        # store the outputs from the conv and pooling layers into vars
        layer_outputs = [layer.output for layer in model.layers[:5]]

        # create a model that will return these outputs, given the model input
        act_model = models.Model(inputs=model.input, outputs=layer_outputs)

        # combine all images to allow visualizaiton of any image from data set
        all_images = np.vstack([train_images, test_images])

        # predictions need more than one image to work, but less than entire set
        if img_num <= len(all_images)-100:

            # therefore, add 100 to the image indexed and pred for 100 images
            acts = act_model.predict(all_images[img_num:(img_num+100)])

        # however, if the image is near the end and would throw an OOB error
        else:

            # just extract everything from the image to the end
            acts = act_model.predict(all_images[img_num:len(all_images)])

        # retrieve the layer in acts based on layer number chosen - 1 (index)
        lyr = acts[lyr_num-1]

        # the image will be the first one since that's where preds started
        img = lyr[0]

        # plot what the intermediate activation looks like for a given filter
        plt.matshow(img[:, :, c_num])

        # specify the image, layer, and channel number of the activation
        plt.title('Channel {} of Activation of Layer {} on Image #{}'.format(c_num, lyr_num, img_num+1))

        # show the plot
        plt.show()

# visualizing the first channel of activation of first layer on the first image
viz_acts(0, 1, 1)


