"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW2.2.2
09/28/2021
"""

# imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
# from tensorflow.keras import activations

# import the Auto MPG dataset

# link for downloading dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

# assign column names
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# read in the dataset using the url and specifying other params for reading
df = pd.read_csv(url, names = column_names, na_values = '?', comment = '\t',
                 sep = ' ', skipinitialspace = True)

# visualizing the data using Dr. Hickman's Seaborn_visualizer.py script
import Seaborn_visualizer as SBV
SBV.get_pd_info(df)
SBV.pd_general_plots(df, HUE = 'Origin')
# SBV.pandas_2D_plots(df, col_to_plot = [1, 4, 5], HUE = 'Origin')

#----------------------------------------
# PRE-PROCESS DATA 
# (EXTRACT AND CONVERT TO TENSOR)
#----------------------------------------

print('----------------------')
print('EXTRACT DATA')
print('----------------------')

# SELECT COLUMNS TO USE AS VARIABLES

# indexing the dependent variables and independent variable to be used 
x_col = [1, 2, 3, 4, 5]
y_col = [0]

# placing the indices into one list
xy_col = x_col + y_col

# converting the list of column indices to dataframe keys
x_keys  = SBV.index_to_keys(df, x_col)   # dependent vars
y_keys  = SBV.index_to_keys(df, y_col)   # independent var
xy_keys = SBV.index_to_keys(df, xy_col)  # independent vars and dependent var

print('X =', x_keys); print('Y =', y_keys)
# SBV.pd_general_plots(df[xy_keys])

# converting the parts of df associated with the variables to numpy
x = df[x_keys].to_numpy()
y = df[y_keys].to_numpy()

# removing nan values if any are present
xtmp = []
ytmp = []
for i in range(0, len(x)):
    if (not 'nan' in str(x[i])):
        xtmp.append(x[i])
        ytmp.append(y[i])
x = np.array(xtmp)
y = np.array(ytmp)

# PARTITION DATA
f_train = 0.8
f_val   = 0.15
f_test  = 0.05
indices = np.random.permutation(x.shape[0])
CUT1 = int(f_train*x.shape[0])
CUT2 = int((f_train + f_val)*x.shape[0]) #print(CUT,x.shape,indices.shape)
train_idx = indices[:CUT1]
val_idx = indices[CUT1:CUT2]
test_idx = indices[CUT2:]
x_train, y_train =  x[train_idx, :], y[train_idx, :]
x_val,   y_val   =  x[val_idx, :], y[val_idx, :]
x_test,  y_test  =  x[test_idx, :], y[test_idx, :]
print('------PARTITION INFO---------')
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# NORMALIZE DATA
# print(np.mean(x_train,axis=0),np.std(x_train,axis=0))
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

# # #PLOT INITIAL DATA
iplot=False
if(iplot):
    fig, ax = plt.subplots()
    FS=18   #FONT SIZE

    for indx in range(0,x_train.shape[1]):
        plt.plot(x_train[:,indx],y_train,'o')
        plt.plot(x_val[:,indx],y_val,'o')
        plt.xlabel(x_keys[indx], fontsize=FS)
        plt.ylabel(y_keys[0], fontsize=FS)
        plt.show(); plt.clf()

# exit()
#-------------------------------------------
# TRAIN WITH KERAS
#-------------------------------------------

# HYPERPARAMETERS 
optimizer = 'rmsprop'
loss_function = 'MeanSquaredError' 
# loss_function="MeanAbsoluteError" 
learning_rate = 0.051
numbers_epochs = 200
model_type = 'linear'
input_shape = (x_train.shape[1],)

# batch_size=1                        # stochastic training
# batch_size=int(len(x_train)/2.)     # mini-batch training
batch_size = len(x_train)             # batch training

# LOGISTIC REGRESSION MODEL
if(model_type == 'logistic'):
    act = 'sigmoid'
    model = keras.Sequential([
    layers.Dense(1,
    activation = act,
    input_shape = (1,)),
    ])

if(model_type == 'linear'):
    model = keras.Sequential([
    layers.Dense(1, activation = 'linear', input_shape = input_shape),
    ])

# LINEAR REGRESSION MODEL
print(model.summary()); #print(x_train.shape,y_train.shape)
print("initial parameters:", model.get_weights())

# COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss_function)

# TRAINING YOUR MODEL
history = model.fit(x_train,
                    y_train,
                    epochs = numbers_epochs,
                    batch_size = batch_size,
                    validation_data = (x_val, y_val))

history_dict = history.history

# make predictions from the train, val, and test data using the model
yp_train = model.predict(x_train)
yp_val   = model.predict(x_val) 
yp_test  = model.predict(x_test)

# UN-NORMALIZE DATA (CONVERT BACK TO ORIGINAL UNITS)
x_train  = x_std*x_train + x_mean 
x_val    = x_std*x_val + x_mean
x_test   = x_std*x_test + x_mean
y_train  = y_std*y_train + y_mean 
y_val    = y_std*y_val + y_mean
y_test   = y_std*y_test + y_mean 
yp_train = y_std*yp_train + y_mean 
yp_val   = y_std*yp_val + y_mean
yp_test  = y_std*yp_test + y_mean

# print(input_shape, x_train.shape, yp_train.shape, y_train.shape)

# PLOT INITIAL DATA

# FUNCTION PLOTS
def plot_1(xcol, xla, yla):
    fig, ax = plt.subplots()
    ax.plot(x_train[:, xcol], y_train, 'o', label = 'Training')
    ax.plot(x_val[:, xcol], y_val, 'x', label = 'Validation')
    ax.plot(x_test[:, xcol], y_test, '*', label = 'Test')
    ax.plot(x_train[:, xcol], yp_train, '.', label = 'Model')
    plt.title(yla + ' Against ' + xla + ' in Multivariate Linear Model')
    plt.xlabel(xla)
    plt.ylabel(yla)
    plt.legend()
    plt.show()
    
iplot = True
if(iplot):
    # FONT SIZE
    # FS = 18
    
    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # calling the function-plotting function for each dependent variable
    for i in range(0, len(x_keys)):
        plot_1(i, x_keys[i], 'MPG')

    # PARITY PLOT
    plt.plot(yp_train, yp_train, 'k--', label = 'Perfect Performance')
    plt.plot(y_train, yp_train, 'o', label = 'Training')
    plt.plot(y_val, yp_val, 'x', label = 'Validation')
    plt.plot(y_test, yp_test, '*', label = 'Test')
    plt.title('Parity Plot of Actual and Predicted MPG')
    plt.xlabel('Predicted MPG')
    plt.ylabel('Actual MPG')
    plt.legend()
    plt.show()
    plt.clf()
    
    """
    # FEATURE DEPENDENCE
    for indx in range(0, x_train.shape[1]):
        #TRAINING
        plt.plot(x_train[:, indx], y_train, 'ro')
        plt.plot(x_train[:, indx], yp_train, 'bx')
        plt.xlabel(x_keys[indx], fontsize = FS)
        plt.ylabel(y_keys[0], fontsize = FS)
        plt.show()
        plt.clf()

        plt.plot(x_val[:, indx], y_val, 'ro')
        plt.plot(x_val[:, indx], yp_val, 'bx')
        plt.xlabel(x_keys[indx], fontsize = FS)
        plt.ylabel(y_keys[0], fontsize = FS)
        plt.show()
        plt.clf()
    """

