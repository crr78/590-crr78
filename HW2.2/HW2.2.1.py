"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW2.2.1
09/28/2021
"""

# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

# user parameters
IPLOT = True
INPUT_FILE = 'planar_x1_x2_y.json'
X_KEYS = ['x1', 'x2']
# INPUT_FILE = 'planar_x1_x2_x3_y.json'
# X_KEYS = ['x1', 'x2', 'x3']
Y_KEYS = ['y']

# number of fitting parameters to use based on number of x and y variables
NFIT = len(X_KEYS) + len(Y_KEYS)

# parameter to decide which type of model to use
# model_type = 'linear'
model_type = 'logistic'

# save history for plotting at the end
epoch = 1        # initialize num of passes through entire training set as 1
epochs = []      # stores the number of passes through the entire training set
loss_train = []  # MSE of truth and pred from train at each optimizer step
loss_val = []    # MSE of truth and pred from val at each optimizer step

# read file
with open(INPUT_FILE) as f:
	my_input = json.load(f)  # read into dictionary

# convert dictionary input into x and y matrices (similar to pandas dfs)   
X = []
Y = []
for key in my_input.keys():
	if(key in X_KEYS): X.append(my_input[key])
	if(key in Y_KEYS): Y.append(my_input[key])

# transpose the matrices
X = np.transpose(np.array(X))
Y = np.transpose(np.array(Y))
print('--------INPUT INFO-----------')
print('X shape:', X.shape)
print('Y shape:', Y.shape, '\n')

# TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
# XMEAN=np.mean(X,axis=0); XSTD=np.std(X,axis=0) 
# YMEAN=np.mean(Y,axis=0); YSTD=np.std(Y,axis=0) 

# NORMALIZE 
# X=(X-XMEAN)/XSTD;  Y=(Y-YMEAN)/YSTD  

# assign the proportions of the data to allocate for train, val, and test sets
f_train = 0.8
f_val   = 0.15
f_test  = 0.05

# check to make sure the proportions sum to a whole
if(f_train + f_val + f_test != 1.0):
	raise ValueError('f_train + f_val + f_test MUST EQUAL 1')

# create indices for partitioning data into train, validation, and test
rand_indices = np.random.permutation(X.shape[0])
CUT1 = int(f_train*X.shape[0]) 
CUT2 = int((f_train + f_val)*X.shape[0]) 
train_idx = rand_indices[:CUT1]
val_idx = rand_indices[CUT1:CUT2]
test_idx = rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

# sigmoid function for mapping input values to points in (0:1) for logistic
def S(x):
    return 1.0/(1.0 + np.exp(-x))

# fit the linear data to the sigmoid function for the logistic model
if model_type == 'logistic':
    Y = S(Y)

# function for modeling using either linear or logistic regression
def model(x,p):

	# use matrix multiplication to generalize for cases with 2+ predictor vars
	linear = p[0] + np.matmul(x, p[1:].reshape(NFIT-1, 1))

	# return the predictions made from the input values and current params
	if model_type == 'linear':   return  linear

	# compare pred to truth (S(Y)) correctly by returning preds via S(x) func  
	if model_type == 'logistic': return  S(linear)

# function to make various predictions for given parameterization
def predict(p):
	global YPRED_T, YPRED_V, YPRED_TEST, MSE_T, MSE_V
	YPRED_T = model(X[train_idx], p)
	YPRED_V = model(X[val_idx], p)
	YPRED_TEST = model(X[test_idx], p)
	MSE_T = np.mean((YPRED_T - Y[train_idx])**2.0)
	MSE_V = np.mean((YPRED_V - Y[val_idx])**2.0)

# loss function for computing train loss given params and specific indexes
def loss(p, index_2_use):
	errors = model(X[index_2_use], p) - Y[index_2_use]  # vector of errors
	training_loss = np.mean(errors**2.0)				 # MSE
	return training_loss

# minimizer func searches for the optimal params that minimize the loss func
def minimizer(f, xi, algo = 'GD', LR = 0.01):
	global epoch, epochs, loss_train, loss_val 

	# parameters
	iteration = 1          # iteration counter
	dx = 0.0001            # step size for finite difference
	max_iter = 5000        # maximum number of iterations
	tol = 10**-10          # exit after change in f is less than this 
	NDIM = len(xi)         # dimension of optimization problem

	# using higher learning rates works much better for logistic model
	if model_type == 'logistic':

		# choose learning rate of 10 if 2 or less dependent vars 
		if len(X_KEYS) <= 2:
			LR = 10

		# else, if more than 2, a learning rate of 5 works better
		else:
			LR = 5

	# optimization loop
	while(iteration<=max_iter):

		# using the "batch" training paradigm
        
        # define the training index to be used before the optimizer begins
		if iteration == 1: index_2_use = train_idx
        
        # optimizer sees all training data for all of the following steps
		if iteration > 1:  epoch += 1
        
        # numerically compute gradient
        
		df_dx=np.zeros(NDIM);   # initialize gradient vector
		for i in range(0,NDIM): # loop over dimensions

			dX=np.zeros(NDIM);  # initialize step array
			dX[i]=dx;           # take set along ith dimension
			xm1=xi-dX;          # step back
			xp1=xi+dX;          # step forward

			# central finite difference
			grad_i = (f(xp1, index_2_use) - f(xm1, index_2_use))/dx/2

			# update gradient vector 
			df_dx[i] = grad_i 
			
		# take an optimzer step
        
        # move x a step in the opp direction from gradient if GD is being used
		if algo == "GD": xip1 = xi - LR*df_dx

		# report and save data for plotting
		if(iteration%1==0):
			predict(xi)	# make prediction for current parameterization
			print(iteration,"	",epoch,"	",MSE_T,"	",MSE_V) 

			# update the number of epochs, train MSE, and validation MSE
            
            # store the current number of epochs
			epochs.append(epoch)
            # record the train MSE of the current optimzer iteration
			loss_train.append(MSE_T)
            # record the validation MSE of the current optimizer iteration
			loss_val.append(MSE_V)

			# calculate MAE between preds from current and previous parameters
			df = np.absolute(f(xip1, index_2_use) - f(xi, index_2_use))
            
            # if MAE is less than tolerance, stop training
			if(df < tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break
        
		# update the parameters to be used for the next step of the optimizer
		xi = xip1
        
        # increment the number of iterations of the optimizer by 1
		iteration += 1
        
    # return the optimal parameters
	return xi

# fit model

# random initial guess for fitting parameters
po = np.random.uniform(2, 1., size = NFIT)

print(X[train_idx].shape)
print(po)
print(po.shape)
print(X[train_idx].shape)
print((po[1:].reshape(NFIT-1, 1)).shape)
print((np.matmul(X[train_idx], po[1:].reshape(NFIT-1, 1))).shape)
exit()

# train the model using optimizer
p_final = minimizer(loss, po)	
print("OPTIMAL PARAM:", p_final)
predict(p_final)

# generate plots

# plot training and validation loss history
def plot_0():
	fig, ax = plt.subplots()
	ax.plot(epochs, loss_train, 'o', label = 'Training loss')
	ax.plot(epochs, loss_val, 'o', label = 'Validation loss')
	plt.title('Training & Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

# parity plot
def plot_1():
	fig, ax = plt.subplots()
	ax.plot(Y[train_idx], YPRED_T, 'o', label = 'Training') 
	ax.plot(Y[val_idx], YPRED_V, 'x', label = 'Validation') 
	ax.plot(Y[test_idx], YPRED_TEST, '*', label = 'Test')
    
	if model_type == 'linear':
		plt.title('Parity Plot of Actual and Predicted Y Data')
		plt.xlabel('Actual Y Data Values')
		plt.ylabel('Predicted Y Data Values')
        
	else:
		plt.title('Parity Plot of Actual and Predicted S(Y) Data')
		plt.xlabel('Actual S(Y) Values')
		plt.ylabel('Predicted S(Y) Values')
	
	plt.legend()
	plt.show()
	
if(IPLOT):
	plot_0()
	plot_1()

	# unnormalize relevant arrays
	# X=XSTD*X+XMEAN 
	# Y=YSTD*Y+YMEAN 
	# YPRED_T=YSTD*YPRED_T+YMEAN 
	# YPRED_V=YSTD*YPRED_V+YMEAN 
	# YPRED_TEST=YSTD*YPRED_TEST+YMEAN 

	# plot_1()

