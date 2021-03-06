"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW5.0: 02-train.py
11/02/2021
"""


#--------
# IMPORTS
#--------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import metrics
from keras.utils import to_categorical


#----------------
# CODE PARAMETERS
#----------------

model_type = 'cnn'
# model_type = 'rnn'


#--------------------------------------------
# LOAD, SHUFFLE, AND SAMPLE PROCESSED DATASET
#--------------------------------------------

# read in the processed data set
processed_data = pd.read_csv('processed_data.csv')

# if an rnn model will be built
if model_type == 'rnn':

    # assign the corpus to a variable
    corpus = list(processed_data['Chunk'])

    # assign the variables to a label
    labels = list(processed_data['Label'])

    # initialize a list to store chunks with less than 10 words
    less_chunks = []

    # initialize a list to store the labels that correspond to less_chunks list
    less_labels = []

    # initialize a counter
    count = 0

    # for each chunk
    for chunk in corpus:

        # if there are no more than 10 words in the chunk
        if len(chunk.split()) <= 10:

            # append that to the list of less chunks
            less_chunks.append(chunk)

            # use the counter to append the corresponding label
            less_labels.append(labels[count])

        # increment the counter
        count += 1

    # assign the old corpus to be the new version with less words
    corpus = np.asarray(less_chunks)

    # assign the old labels to be the new version with less labels
    labels = np.asarray(less_labels).astype('float32')

else:

    # assign the corpus to use as the 'Chunk' column and convert to array
    corpus = np.asarray(processed_data['Chunk'])

    # designate the labels to use as the 'Label' column and convert to list
    labels = np.asarray(processed_data['Label']).astype('float32')

# shuffle the indices to create a random sample
rand_indices = np.random.RandomState(seed=0).permutation(corpus.shape[0])

# designate a fraction of the data set to use to circumvent memory issues
frac = 0.1

# specify the index corresponding to the cut-off
CUT = int(frac*corpus.shape[0])

# limit the random indices to be only those up to the index of the cut
rand_idx = rand_indices[:CUT]

# reassign the corpus and labels to be the shuffled, limited version of original
corpus, labels = list(corpus[rand_idx]), list(labels[rand_idx])


#------------------------------------------
# FORM DICTIONARY WITH WORD INDICE MAPPINGS
#------------------------------------------

# function for converting the corpus to a dict mapping each word to unique idx
def form_dictionary(samples):
    
    # initialize the dictionary for storing the words and indices
    token_index = {};
    
    # for each chunk
    for sample in samples:
        
        # for each word in the current chunk
        for word in sample.split():
            
            # if the word has not already been assigned to dict
            if word not in token_index:
                
                # assign the word and record the index
                token_index[word] = len(token_index) + 1

    # initialize a list for storing the transformed text
    transformed_text=[]
    
    # for each chunk
    for sample in samples:
        
        # initialize a list for storing how the new indice-mapped text looks
        tmp=[]
        
        # for each word in the current chunk
        for word in sample.split():
            
            # append the word's index to the list
            tmp.append(token_index[word])
            
        # append the list to the main list
        transformed_text.append(tmp)

    # print("CONVERTED TEXT:", transformed_text)
    # print("VOCABULARY-2 (SKLEARN): ",token_index)
    
    # return token index along with indice-mapped chunks
    return [token_index, transformed_text]

# designate vars for the token index and indice-mapped chunks
[vocab, x] = form_dictionary(corpus)


#-----------------------------------
# ONE-HOT-ENCODE THE DATA AND LABELS
#-----------------------------------

# CHOLLET: LISTING 6.1: WORD-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
# function for one-hot-encoding each chunk of words
def one_hot_encode(samples):
    
    # define the maximum number of words to use per chunk
    max_length = 100

    # initialize the matrix to represent each one-hot-encoded chunk
    results = np.zeros(shape=(len(samples), max_length, max(vocab.values())+1))

    # for each chunk and corresponding index
    for i, sample in enumerate(samples):

        # for each word and corresponding index
        for j, word in list(enumerate(sample.split()))[:max_length]:

            # retrieve the index of the word in the vocab dictionary
            index = vocab.get(word)

            # assign 1 in column (dict index #) of row (word) of chunk (sample)
            results[i, j, index] = 1.
    
    # the first column in each sample is all zeros, so disregard it
    results=results[:,:,1:]

    # return the one-hot-encoded chunks
    return results

# CHOLLET; IMDB (CHAPTER-3: P69)
# function for one-hot-encoding each sequence
def vectorize_sequences(sequences, dimension):

    # initialize a matrix of zeros
    results = np.zeros((len(sequences), dimension))

    # for each sequence and its index
    for i, sequence in enumerate(sequences):

        # set specific indices of the result of the current index to 1
        results[i, sequence] = 1.

    # return the vectorized sequences
    return results

# one-hot-encode each chunk according to the model being used
if model_type == 'rnn':
    x_ohe = vectorize_sequences(x, len(vocab)+1)
else:
    x_ohe = one_hot_encode(corpus)

# designate a var for storing the vectorized version of the labels
labels_array = to_categorical(labels)


#----------------
# HYPERPARAMETERS
#----------------

nodes_in = 32                            # number of input layer nodes
nodes_hidden = 32                        # number of hidden layer nodes
nodes_out = 3                            # number of output layer nodes
ks = 3                                   # kernel size
ps = 2                                   # pool size 
act_in = 'relu'                          # activation function for input layer
act_hidden = 'relu'                      # activation function for hidden layer
act_out = 'softmax'                      # activation function for output layer
shape_in = x_ohe.shape[1:]               # shape of input for input layer
ki = 'ones'                              # specify kernel initializer
bi = 'zeros'                             # specify bias initializer
opt = 'rmsprop'                          # optimizer to use
loss_func = 'categorical_crossentropy'   # loss function to use
metrics = ['accuracy', metrics.AUC()]    # use accuracy, AUC for model metrics
num_epochs = 20                          # number of epochs
b_size = 32                              # batch size


#------------
# BUILD MODEL
#------------

# function for building 1D-CNN
def build_cnn():

    # create an instance of a Keras Sequential model
    model = models.Sequential()

    # incorporate an instance of a Conv1D layer and specify input parameters
    model.add(layers.Conv1D(nodes_in, ks, activation=act_in,
                            input_shape=shape_in, kernel_initializer=ki,
                            bias_initializer = bi))

    # include a dropout layer for regularization
    model.add(layers.Dropout(0.5))

    # reshape array for correct input into the DFF
    model.add(layers.Flatten())

    # specify dff layers, nodes, and activation functions
    model.add(layers.Dense(nodes_hidden, activation=act_hidden))
    model.add(layers.Dense(nodes_out, activation=act_out))

    # print summary of model
    model.summary()

    # return the model
    return model

# function for building RNN model
def build_rnn():

    # create an instance of a Keras Sequential model
    model = models.Sequential()

    # add an embedding layer to map word indices in dictionary to dense vectors
    model.add(layers.Embedding(len(vocab.keys()), nodes_in))

    # incorporate an instance of a simple RNN layer and specify input parameters
    model.add(layers.SimpleRNN(nodes_in))

    # include a dropout layer for regularization
    model.add(layers.Dropout(0.5))

    # specify output layer nodes, activation function
    model.add(layers.Dense(nodes_out, activation=act_out))

    # prints summary of model
    model.summary()

    # return the model
    return model

# build cnn or rnn based on choice of model type
if model_type == 'cnn':
    model = build_cnn()
else:
    model = build_rnn()


#---------------
# PARTITION DATA
#---------------

# specify fraction of data set to use for training
f_train = 0.8

# specify fraction of data set to use for validating
f_val   = 0.15

# shuffle the indices of the data set
rand_indices = np.random.RandomState(seed=0).permutation(x_ohe.shape[0])

# specify the index where the training data ends
CUT1 = int(f_train*x_ohe.shape[0]);

# specify the index where the validation data will end
CUT2 = int((f_train+f_val)*x_ohe.shape[0]);

# use the cut to generate a random training index
train_idx = rand_indices[:CUT1]

# use the cut to generate a random validation index
val_idx   = rand_indices[CUT1:CUT2]

# use the cut to generate a random testing index
test_idx  = rand_indices[CUT2:]

# use the indices to create a random set of training data and labels
x_train, y_train = x_ohe[train_idx, :], labels_array[train_idx]

# use the indices to create a random set of validation data and labels
x_val, y_val = x_ohe[val_idx, :], labels_array[val_idx]

# use the indices to create a random set of testing data and labels
x_test, y_test = x_ohe[test_idx, :], labels_array[test_idx]

# print the partition information
print('------PARTITION INFO---------')
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


#----------------------
# COMPILE & TRAIN MODEL
#----------------------

# compile model, specifying optimizer, loss, and evaluation metrics to use
model.compile(optimizer=opt, loss=loss_func, metrics=metrics)

# fit the model using the train set, validate the model using the validation set
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=b_size,
                    validation_data=(x_val, y_val), verbose=0)


#---------------
# RECORD RESULTS
#---------------

# record categorical cross-entropy, accuracy, and AUC on the train and val sets
train_cce = history.history['loss']          # training loss
val_cce   = history.history['val_loss']      # validation loss
train_acc = history.history['accuracy']      # training accuracy
val_acc   = history.history['val_accuracy']  # validation accuracy
train_auc = history.history['auc']           # training AUC
val_auc   = history.history['val_auc']       # validation AUC

# record the minimum categorical cross-entropy, maximum accuracy, and AUC for 
# both train and val, as well as the epochs that each of the extremes occur at
min_train_cce = np.min(train_cce)
min_val_cce = np.min(val_cce)
min_train_cce_epoch = np.argmin(train_cce) + 1
min_val_cce_epoch = np.argmin(val_cce) + 1
max_train_acc = np.max(train_acc)
max_val_acc = np.max(val_acc)
max_train_acc_epoch = np.argmax(train_acc) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1
max_train_auc = np.max(train_auc)
max_val_auc = np.max(val_auc)
max_train_auc_epoch = np.argmax(train_auc) + 1
max_val_auc_epoch = np.argmax(val_auc) + 1

# print results
print('-------Results-------')
print('The minimum training categorical cross-entropy is', 
      round(min_train_cce, 2), 'and it occurs at epoch #', min_train_cce_epoch)
print('The minimum validation categorical cross-entropy is', 
      round(min_val_cce, 2), 'and it occurs at epoch #', min_val_cce_epoch)
print('The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch))
print('The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch), '\n')
print('The maximum training AUC is {} and it occurs at epoch # {}.'.format(round(max_train_auc, 4), max_train_auc_epoch))
print('The maximum validation AUC is {} and it occurs at epoch # {}.'.format(round(max_val_auc, 4), max_val_auc_epoch), '\n')


#------------------
# VISUALIZE RESULTS
#------------------

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


#---------------------
# SAVE MODEL & RESULTS
#---------------------

# save model
if model_type == 'cnn':
    model.save('cnn_mod.h5')
else:
    model.save('rnn_mod.h5')

# results
line1 = '-------Results-------' + '\n'

line2 = 'The minimum training categorical cross-entropy is ' +  str(round(min_train_cce, 2)) + ' and it occurs at epoch #' +  str(min_train_cce_epoch) + '\n'

line3 = 'The minimum validation categorical cross-entropy is ' + str(round(min_val_cce, 2)) + ' and it occurs at epoch #' + str(min_val_cce_epoch) + '\n'

line4 = 'The maximum training accuracy is {}% and it occurs at epoch # {}.'.format(round(max_train_acc*100, 2), max_train_acc_epoch) + '\n'

line5 = 'The maximum validation accuracy is {}% and it occurs at epoch # {}.'.format(round(max_val_acc*100, 2), max_val_acc_epoch) + '\n'

line6 = 'The maximum training AUC is {} and it occurs at epoch # {}.'.format(round(max_train_auc, 4), max_train_auc_epoch) + '\n'

line7 = 'The maximum validation AUC is {} and it occurs at epoch # {}.'.format(round(max_val_auc, 4), max_val_auc_epoch) + '\n'

# write cnn results
if model_type == 'cnn':
    results_file = open('log_cnn.txt', 'w')
    results_file.write(line1)
    results_file.write(line2)
    results_file.write(line3)
    results_file.write(line4)
    results_file.write(line5)
    results_file.write(line6)
    results_file.write(line7)
    results_file.close()

# write rnn results
else:
    results_file = open('log_rnn.txt', 'w')
    results_file.write(line1)
    results_file.write(line2)
    results_file.write(line3)
    results_file.write(line4)
    results_file.write(line5)
    results_file.write(line6)
    results_file.write(line7)
    results_file.close()


