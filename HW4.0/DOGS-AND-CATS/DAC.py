"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW4.0: Dogs and Cats
10/20/2021
"""

#--------
# IMPORTS
#--------

from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import shutil
import zipfile


#--------------------------------------
# GET ZIP FILE FROM LINK AND UNCOMPRESS
#--------------------------------------

# link where data set is stored
link = 'https://chevyrobertsonanalytics.georgetown.domains/Train.zip'

# use requests library with link
get_rq = requests.get(link)

# get the content from the zip file
zip_file = zipfile.ZipFile(io.BytesIO(get_rq.content))

# unzip the file and store folder into a directory
zip_file.extractall('kaggle_original_data')


#----------------------------------------------------------
# COPY IMAGES TO TRAINING, VALIDATION, AND TEST DIRECTORIES
#----------------------------------------------------------

# path to the directory where the original dataset was uncompressed
original_dataset_dir = 'kaggle_original_data/Train'

# make a directory for storing the smaller dataset
base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)

# make directories for the training, validation, and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# make a directory to store the training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# make a directory to store the training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# make a directory to store the validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# make a directory to store the validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# make a directory to store the testing cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# make a directory to store the testing dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# copy the first 1,000 cat images to the training cats directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# copy the next 500 cat images to the validation cats directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# copy the next 500 cat images to the testing cats directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# copy the first 1,000 dog images to the training dogs directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copy the next 500 dog images to the validation dogs directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copy the next 500 dog images to the testing dogs directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# as a sanity check, counting the number of pictures in each training split
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

#---------------------------------------------------------------
# INSTANTIATING A SMALL CONVNET FOR DOGS VS. CATS CLASSIFICATION
#---------------------------------------------------------------

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# viewing how the dims of the feature maps change with every successive layer
print(model.summary())


#-----------------------------------
# CONFIGURING THE MODEL FOR TRAINING
#-----------------------------------

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


#---------------------------------------------------------
# USING ImageDataGenerator TO READ IMAGES FROM DIRECTORIES
#---------------------------------------------------------

# rescaling all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# read the images from the specified directory and store in a generator object
train_generator = train_datagen.flow_from_directory(train_dir,  # target dir

                                                    # resizes images to 150x150
                                                    target_size=(150, 150),
                                                    
                                                    # batch size
                                                    batch_size=20,

                                                    # use binary labels
                                                    class_mode='binary')

# repeat for validation set
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')


#------------------------------------------
# FITTING THE MODEL USING A BATCH GENERATOR
#------------------------------------------

# since data is in batches, fit_generator is used instead of "fit" to fit model

# steps per epoch and val steps are num samples to draw from gen b4 epoch ends
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)


#-----------------
# SAVING THE MODEL
#-----------------

# saves model (including configuration, weights, and optimizer) to HDF5 file
model.save('cats_and_dogs_small_1.h5')


#-------------------------------------------------------
# DISPLAYING CURVES OF LOSS AND ACCURACY DURING TRAINING
#-------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#--------------------------------------------------------------------
# SETTING UP A DATA AUGMENTATION CONFIGURATION VIA ImageDataGenerator
#--------------------------------------------------------------------

datagen = ImageDataGenerator(rescale = 1./255,         # normalizes 
                             rotation_range=40,        # range of rotation
                             width_shift_range=0.2,    # range of horiz shift
                             height_shift_range=0.2,   # range of vert shift
                             shear_range=0.2,          # shearing transformation
                             zoom_range=0.2,           # zooming inside image
                             horizontal_flip=True,     # flip image horizontally
                             fill_mode='nearest')      # fills new pixels

# join the filename and directory to form path for each train cat image
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# select an image to augment
img_path = fnames[3]

# read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# conver the image to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# generates batches of randomly transformed images, stops after 4 images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()


#---------------------------------------------
# DEFINING A NEW CONVNET THAT INCLUDES DROPOUT
#---------------------------------------------

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # dropout layer sets inputs to 0 at 0.5 rate
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


#--------------------------------------------------------------------
# TRAINING THE CONVNET USING DATA AUGMENTATION GENERATORS AND DROPOUT
#--------------------------------------------------------------------

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)


#-----------------
# SAVING THE MODEL
#-----------------

model.save('cats_and_dogs_small_2.h5')


#-------------------------------------------------------
# DISPLAYING CURVES OF LOSS AND ACCURACY DURING TRAINING
#-------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#---------------------------
# USING A PRETRAINED CONVNET
#---------------------------


#-------------------------------------------
# INSTANTIATING THE VGG16 CONVOLUTIONAL BASE
#-------------------------------------------

# develop convolutional base using VGG16 constructor from Keras
conv_base = VGG16(weights='imagenet',         # initialize weights from imagenet
                  include_top=False,          # whether to include imagenet dcc
                  input_shape=(150, 150, 3))  # shape of image tensors fed to NN

print(conv_base.summary())


#------------------------------------------------------------
# EXTRACTING FEATURES USING THE PRETRAINED CONVOLUTIONAL BASE
#------------------------------------------------------------

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# function for fast feature extraction without data augmentation
def extract_features(directory, sample_count):

    # initialize a tensor for storing the features to be extracted from the imgs
    features = np.zeros(shape=(sample_count, 4, 4, 512))

    # initialize an array for storing the labels of the images
    labels = np.zeros(shape=(sample_count))

    # instantiating a generator to be used for extracting the images and labels
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    
    # initializing counter to use for indexing the batches
    i = 0

    # for the sample of images and sample of labels in the generator object
    for inputs_batch, labels_batch in generator:

        # use the conv_base model to predict features given input images
        features_batch = conv_base.predict(inputs_batch)

        # assemble the features into batches
        features[i * batch_size : (i + 1) * batch_size] = features_batch

        # assemble the corresponding labels into batches
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        # increment counter
        i += 1

        # break the loop after every image has been seen once
        if i * batch_size >= sample_count:
            break

    # return the extracted features and corresponding labels
    return features, labels

# perform feature extraction (with labels) for train, val, and test
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# flatten the extracted features so that they can be interpreted by dcc properly
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))


#-------------------------------------------------------
# DEFINING AND TRAINING THE DENSELY CONNECTED CLASSIFIER
#-------------------------------------------------------

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                    validation_data=(validation_features, validation_labels))


#-------------------------------------------------------
# DISPLAYING CURVES OF LOSS AND ACCURACY DURING TRAINING
#-------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#-------------------------------------
# VISUALIZING INTERMEDIATE ACTIVATIONS
#-------------------------------------


#-----------
# LOAD MODEL
#-----------

model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())


#-----------------------------
# PREPROCESSING A SINGLE IMAGE
#-----------------------------

# specify the path of the image to preprocess
img_path = 'cats_and_dogs_small/test/cats/cat.1700.jpg'

# load the image and resize to 150x150
img = image.load_img(img_path, target_size=(150, 150))

# convert the image to a Numpy array
img_tensor = image.img_to_array(img)

# expand the shape of the array, specifying pos. of new axis in expanded shape
# now this is a 4D tensor
img_tensor = np.expand_dims(img_tensor, axis=0)

# normalizing
img_tensor /= 255.

# shape is now (1, 150, 150, 3)
print(img_tensor.shape)


#----------------------------
# DISPLAYING THE TEST PICTURE
#----------------------------

plt.imshow(img_tensor[0])
plt.show()


#------------------------------------------------------------------------
# INSTANTIATING A MODEL FROM AN INPUT TENSOR AND A LIST OF OUTPUT TENSORS
#------------------------------------------------------------------------

# store the outputs from the conv and pooling layers into vars
layer_outputs = [layer.output for layer in model.layers[:8]]

# create a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


#----------------------------------
# RUNNING THE MODEL IN PREDICT MODE
#----------------------------------

# returns a list of five Numpy arrays; one array per layer activation
activations = activation_model.predict(img_tensor)

# activation of the first convolution layer for the cat image input
first_layer_activation = activations[0]

# print the shape of the activation
print(first_layer_activation.shape)


#-------------------------------
# VISUALIZING THE FOURTH CHANNEL
#-------------------------------

# shows fourth channel of the activation of the first layer of the test cat pic 
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()


#--------------------------------
# VISUALIZING THE SEVENTH CHANNEL
#--------------------------------

# shows 7th channel of the activation of the first layer of the test cat pic 
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()


#-----------------------------------------------------------
# VISUALIZING EVERY CHANNEL IN EVERY INTERMEDIATE ACTIVATION
#-----------------------------------------------------------

# initialize a list to store the name of each layer
layer_names = []

# use a for loop to append the name of each layer to the layer_names list
for layer in model.layers[:8]:
    layer_names.append(layer.name)

# define the number of images to chow in each row
images_per_row = 16

# displays the feature maps
for layer_name, layer_activation in zip(layer_names, activations):

    # number of features in the feature map
    n_features = layer_activation.shape[-1]

    # the feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]
    
    # tiles the activation channels in the current matrix
    n_cols = n_features // images_per_row

    # initializing a grid to use for arranging the activations
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # tiles each filter into a big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            
            # specify channel image to show based on row and col index of grid
            channel_image = layer_activation[0, :, :, col*images_per_row + row]

            # post-process the feature to make it visually patable

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128

            # limit the values in the array from 0 to 255, recast type as int
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
          
            # displays the grid
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # develop a scale based on rows/cols of feature map
    scale = 1./size

    # use the scale to scale the rows and cols of the grid
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))

    # title the plot based on the name of the layer being plotted
    plt.title(layer_name)

    # do not plot grid lines
    plt.grid(False)

    # show the plot of the grid of activations for the current layer
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

