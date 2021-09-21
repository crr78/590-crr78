"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW2.1, Part-II: Coding Assignment (Regression)
09/21/2021
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import json

# user parameters
IPLOT = True
INPUT_FILE = 'weight.json'
DATA_KEYS = ['x','is_adult','y']

# define vars establishing initial guess size, xdata, and ydata, respectively
NFIT = 4
xcol = 1
ycol = 2

# read file
with open(INPUT_FILE) as f:
	my_input = json.load(f)  # read into dictionary

# convert input into one large matrix (similar to pandas df)
X = []
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

# transpose the matrix
X = np.transpose(np.array(X))

# select columns for training 
x = X[:,xcol]
y = X[:,ycol]

# compute mean and std of x and y before partitioning and save for later
XMEAN = np.mean(x)
XSTD  = np.std(x)
YMEAN = np.mean(y)
YSTD  = np.std(y)

# normalize
x = (x-XMEAN)/XSTD
y = (y-YMEAN)/YSTD

# create indices for partitioning data into train, validation, and test
f_train = 0.8
f_val   = 0.1
f_test  = 0.1
rand_indices = np.random.permutation(x.shape[0])
CUT1 = int(f_train*x.shape[0])
CUT2 = int((f_train + f_val)*x.shape[0]) 
train_idx = rand_indices[:CUT1]
val_idx   = rand_indices[CUT1:CUT2]
test_idx  = rand_indices[CUT2:]
x_train = x[train_idx]
y_train = y[train_idx]
x_val   = x[val_idx]
y_val   = y[val_idx]
x_test  = x[test_idx]
y_test  = y[test_idx]

# logistic regression model
def model(x,p):
    return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

# save history for plotting at the end
epoch = 0        # initialize num of passes through entire training set as 0
epochs = []      # stores the number of passes through the entire training set
loss_train = []  # MSE of truth and pred from train at each optimizer step
loss_val = []    # MSE of truth and pred from val at each optimizer step

# loss function for computing train loss given params and specific indexes
def loss(p, xb, yb):

    # make preds using the model with train index passed and current params
	yp = model(xb, p)
    
    # compute train mse using predicted and actual weights
	training_loss = (np.mean((yp - yb)**2.0))
    
    # return result of loss function on train to continue updating parameters
	return training_loss

# optimizer searches for the optimal params that minimize the loss function
def optimizer(f, xi, algo = 'GD', LR = 0.001, method = 'batch'):
    
    # allow modification of these variables outside of the function scope
    global epochs, loss_train, loss_val, epoch
    
    # parameters
    dx   = 0.001    # step size for finite difference
    t    = 0        # initial iteration counter
    tmax = 100000   # max number of iterations
    tol  = 10**-10  # exit after change in f is less than this
    NDIM = 4        # number of parameters to optimize
    
    # keep running the code below 'til the max number of iterations is reached
    while(t <= tmax):
        
        # partition training data based on chosen partitioning method
        
        # if paritioning method is batch, use all train data at each iteration
        if method == 'batch':
            # only if the optimizer has not taken a step yet
            if t == 0:
                # define the training index to be used
                idx = train_idx
            # else, if the optimizer has taken at least one step
            else:
                # make preds from train set using model and updated params
                yhat_t = model(x[train_idx], xi)
                # make preds from val set using model and updated params
                yhat_v = model(x[val_idx], xi)
                # compute train mse using predicted and actual weights
                loss_t = np.mean((yhat_t - y[train_idx])**2)
                # compute val mse using predicted and actual weights
                loss_v = np.mean((yhat_v - y[val_idx])**2)
                # store the train MSE of the current iteration into loss_train
                loss_train.append(loss_t)
                # store the val MSE of the current iteration into loss_val
                loss_val.append(loss_v)
                # after 1 iter, optimizer has now seen all training examples
                epoch += 1
                # store the current number of epochs
                epochs.append(epoch)
                
        # if method is mini-batch, use half of the train set at each iteration
        if method == 'mini-batch':
            # only if the optimizer has not taken a step yet
            if t == 0:
                # sample half of train to define the indexes of first batch
                batch1 = np.random.choice(train_idx,  # sample from train
                                          size = int(len(train_idx)/2),  # 0.5
                                          replace = False)  # do not replace
                # use the unsampled indexes to define the indexes of batch 2
                batch2 = np.array([i for i in train_idx if not (i in batch1)])
                # use the indexes of batch1 for the first optimizer iteration
                idx = batch1
            # if the number of optimizer iterations is >0 and odd
            elif t%2 == 1:
                # use the indexes of batch2 for the next optimizer iteration
                idx = batch2
            # else, if the number of optimizer iterations is >0 and even
            else:
                # use the indexes of batch2 for the next optimizer iteration
                idx = batch1
                # make preds from train set using model and updated params
                yhat_t = model(x[train_idx], xi)
                # make preds from val set using model and updated params
                yhat_v = model(x[val_idx], xi)
                # compute train mse using predicted and actual weights
                loss_t = np.mean((yhat_t - y[train_idx])**2)
                # compute val mse using predicted and actual weights
                loss_v = np.mean((yhat_v - y[val_idx])**2)
                # store the train MSE of the current iteration into loss_train
                loss_train.append(loss_t)
                # store the val MSE of the current iteration into loss_val
                loss_val.append(loss_v)
                # after 2 iters, optimizer has now seen all training examples
                epoch += 1
                # store the current number of epochs
                epochs.append(epoch)
                
        # if method is stochastic, use 1 data point from train at each step
        if method == 'stochastic':
            # only if the optimizer has not taken a step yet
            if t == 0:
                # a larger step size tended to work better for stochastic
                dx = 0.01
                # initialize a variable to use for indexing the train set
                j = 0
                # use first index in train set for the first optimizer step
                idx = np.array(train_idx[j])
            # for the remaining optimizer iterations
            else:
                # if we have not reached the last index in train set yet
                if j != (len(train_idx) - 1):
                    # increment the index into train by 1
                    j += 1
                    # use index indexed by j in train set for next iteration
                    idx = np.array(train_idx[j])
                # else, if we have reached the last index in train set
                else:
                    # reset the index into the train set to 0
                    j = 0
                    # use first index in train for the next optimizer step
                    idx = np.array(train_idx[j])
                    # make preds from train set using model and updated params
                    yhat_t = model(x[train_idx], xi)
                    # make preds from val set using model and updated params
                    yhat_v = model(x[val_idx], xi)
                    # compute train mse using predicted and actual weights
                    loss_t = np.mean((yhat_t - y[train_idx])**2)
                    # compute val mse using predicted and actual weights
                    loss_v = np.mean((yhat_v - y[val_idx])**2)
                    # store the train MSE of the current iteration into loss_train
                    loss_train.append(loss_t)
                    # store the val MSE of the current iteration into loss_val
                    loss_val.append(loss_v)
                    # after 200 iters, optimizer has seen all train examples
                    epoch += 1
                    # store the current number of epochs
                    epochs.append(epoch)

        # numerically compute gradient 
        df_dx = np.zeros(NDIM)
        for i in range(0, NDIM):
            dX = np.zeros(NDIM)
            dX[i] = dx 
            xm1 = xi - dX
            df_dx[i] = (f(xi, x[idx], y[idx]) - f(xm1, x[idx], y[idx]))/dx
        
        # if gradient descent is the optimizer being used
        if algo == 'GD':
            # move x a step in the opposite direction from the gradient
            xip1 = xi - LR*df_dx
            
        # if gradient descent + momentum is the optimizer being used:
        if algo == 'GD+M':
            # if this is the first iteration
            if t == 0:
                # initialize exponential decay factor
                alpha = 0.1
                # initialize previous step to zero since this is first step
                previous_step = 0
            # consider past gradients as well when updating x
            xip1 = xi - LR*df_dx - alpha*LR*previous_step
            # store current gradient vector to be used as past gv next time
            previous_step = df_dx
        
        # calculate MAE between preds from current and previous parameters
        df = np.mean(np.absolute(f(xip1,x[idx],y[idx]) - f(xi,x[idx],y[idx])))
            
        # if MAE is less than tolerance, stop training
        if(df < tol):
            print("STOPPING CRITERION MET (STOPPING TRAINING)")
            break
            
        # update parameters for the next iteration of loop
        xi = xip1
        
        # increment the number of iterations of the optimizer by 1
        t += 1
    
    # return the optimal parameters
    return xi

# initial guess for optimizer
po = np.random.uniform(0.1, 1, size = NFIT)

# train model using optimizer and store the optimal params in a variable
popt = optimizer(loss, po, algo = 'GD+M', method = 'mini-batch')
print("OPTIMAL PARAM:", popt)

# use the model to make predictions with the training set
xm_train = np.array(sorted(x_train))
yp_train = np.array(model(xm_train, popt))

# use the model to make predictions with the validation set
xm_val = np.array(sorted(x_val))
yp_val = np.array(model(xm_val, popt))

# use the model to make predictions with the testing set
xm_test = np.array(sorted(x_test))
yp_test = np.array(model(xm_test, popt))

# un-normalize
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

# function plots
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(x_train), unnorm_y(y_train), 'o', label = 'Training set')
	ax.plot(unnorm_x(x_val), unnorm_y(y_val), 'x', label = 'Validation set')
	ax.plot(unnorm_x(x_test), unnorm_y(y_test), '*', label = 'Testing set')
	ax.plot(unnorm_x(xm_train), unnorm_y(yp_train), '-', label = 'Model')
	plt.title('Logistic Model Predicting Customer Weight')
	plt.xlabel('Age [years]')
	plt.ylabel('Weight [lbs]')
	plt.legend()
	plt.show()

# parity plots
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(x_train, popt), y_train, 'o', label = 'Training set')
	ax.plot(model(x_val, popt), y_val, 'o', label = 'Validation set')
	plt.title('Parity Plot of Actual and Predicted Weight')
	plt.xlabel('Normalized Predicted Weight [lbs]')
	plt.ylabel('Normalized Actual Weight [lbs]')
	plt.legend()
	plt.show()

# monitor training and validation loss 
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(epochs, loss_train, 'o', label = 'Training loss')
	ax.plot(epochs, loss_val, 'o', label = 'Validation loss')
	plt.title('Training & Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
