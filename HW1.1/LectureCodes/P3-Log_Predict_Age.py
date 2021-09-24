

"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW1.1, Part-III: Regression Workflow (Logistic (Predicting Age Given Weight))
09/07/2021
"""

#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

# import json package to enable loading of the json data
import json

# import pandas to form data structures more convenient for plotting purposes
import pandas as pd

# import matplotlib to make visualizaitons of the data
import matplotlib.pyplot as plt

# import numpy to perform mathematical operations on arrays as needed
import numpy as np

# import StandardScaler to standardize input variables prior to training
from sklearn.preprocessing import StandardScaler

# import minimize from scipy.optimize to minimize the loss function
from scipy.optimize import minimize

# import train_test_split from sklearn to split data into train and validation
from sklearn.model_selection import train_test_split


#----------------------------------------------------------------------------#
# "Data" Class
#----------------------------------------------------------------------------#

# class for reading, partitioning, and visualizing data and model results
class Data:
    
    # init method reads and partitions the data
    def __init__(self, dataset=json.load(open('weight.json'))):
        
        # initializing the x-axis label
        self.xlab = dataset['xlabel'] + ' [years]'
        
        # initializing the y-axis label
        self.ylab = dataset['ylabel'] + ' [lb]'
        
        # initializing the set of age classifications from the json data
        self.age_class = dataset['is_adult']
        
        # initializing the set of ages from the customer json data
        self.ages = dataset['x']
        
        # initializing the set of weights from the customer json data
        self.weights = dataset['y']
        
        # assemble a dataframe to use for plotting
        self.df = pd.DataFrame({'Age': self.ages, 
                                'Weight': self.weights,
                                'Age_Class': self.age_class})
    
    # plot_loss method for graphing train and val loss per optimizer iteration
    def plot_loss(self, iterations, loss_train, loss_val):
        
        # plot the train loss as a function of num of optimizer iterations
        plt.plot(iterations, loss_train, 'g.', label = 'Training Loss')
        
        # plot the val loss as a function of num of optimizer iterations
        plt.plot(iterations, loss_val, 'r.', label = 'Validation Loss')
        
        # add a title
        plt.title('Training & Validation Loss for Logistic Age(Weight) Model')
        
        # specify current number of optimizer iterations on the x-axis
        plt.xlabel('Number of Optimizer Iterations')
        
        # specify both training and validation loss on the y-axis
        plt.ylabel('Loss')
        
        # add a grid
        plt.grid()
        
        # add a legend to differentiate training loss from validation loss
        plt.legend()
        
        # show the plot
        plt.show()
    
    # plot_model method for better analyzing model results with actual data
    def plot_model(self,
                   x_train_orig,
                   y_train_orig,
                   x_val_orig,
                   y_val_orig,
                   customer_df,
                   y_hat_all_orig):
        
        # the weights need to be sorted for the plot to come out correctly
        non_sorted_df = pd.DataFrame({'Weight': customer_df,
                                      'Age_Class': y_hat_all_orig})

        # sorting df of weight and age class preds by weight column only
        sorted_df = non_sorted_df.sort_values('Weight')
        
        # plot age class as a function of weight using the values in train set
        plt.plot(x_train_orig, y_train_orig, 'g.', label = 'Training Set')
        
        # plot age class as a function of weight using the values in val set
        plt.plot(x_val_orig, y_val_orig, 'rx', label = 'Validation Set')
        
        # plot model by plotting age class preds as func of actual weights
        plt.plot(sorted_df['Weight'],    # sorted weights
                 sorted_df['Age_Class'], # corresponding age class predictions
                 'k-',
                 label = 'Model')
        
        # add a title
        plt.title('Logistic Model Predicting Customer Age Class')
        
        # specify weight on the x-axis
        plt.xlabel('Weight [lb]')
        
        # specify age classification on the y-axis
        plt.ylabel('Age Class (CHILD=0, ADULT=1)')
        
        # add a grid
        plt.grid()
        
        # add a legend to identify the model and differentiate train and val
        plt.legend()
        
        # show the plot
        plt.show()
    
    # plot_parity method for checking congruity of pred vs. actual age class
    def plot_parity(self,
                    y_train_orig,
                    y_hat_train_orig,
                    y_val_orig,
                    y_hat_val_orig):
        
        # create data that would resemble the trend for perfect prediction
        perf_hat = range(0, 2, 1)
        
        # plot pred age class as function of actual age class using train set
        plt.plot(y_train_orig, y_hat_train_orig, 'g.', label = 'Training Set')
        
        # plot pred age class as function of actual age class using val set
        plt.plot(y_val_orig, y_hat_val_orig, 'rx',  label = 'Validation Set')
        
        # plot the line the points would fall on for a perfect model
        plt.plot(perf_hat, perf_hat, 'k--', label = 'Perfect Performance')
        
        # add a title
        plt.title('Parity Plot of Actual and Predicted Age Class (Logistic)')
        
        # specify actual age classification values on the x-axis
        plt.xlabel('Actual Age Class (CHILD=0, ADULT=1)')
        
        # specify predicted age classification on the y-axis
        plt.ylabel('Predicted Age Class (CHILD=0, ADULT=1)')
        
        # add a grid
        plt.grid()
        
        # add a legend to differentiate train from val and identify trend line
        plt.legend()
        
        # show the plot
        plt.show()  

        
#----------------------------------------------------------------------------#
# Preparing the Data
#----------------------------------------------------------------------------#                

# create an instance of the Data class
data_obj = Data()

# use the instance to assign the dataframe of customer data to a variable
customer_df = data_obj.df

# drop age column since we're only using actual weight to predict age class
customer_df = customer_df.drop('Age', axis = 1)

# normalize the input and output variables using StandardScaler()
scaled = StandardScaler().fit_transform(customer_df)

# save the scaled input and output variables to new variables x and y
x, y = scaled[:, 0], scaled[:, 1]

# split the scaled input and output values into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, 
                                                  y, 
                                                  test_size = 0.2, 
                                                  shuffle = True,
                                                  random_state = 0)


#----------------------------------------------------------------------------#
# Defining a Logistic Model Function
#----------------------------------------------------------------------------#

# define a logistic model function that inputs x and p vectors and outputs y
def model_func(x, p):
    
    # return age class given input weight vector and current parameters
    return p[0]/(1 + 1/np.exp((x - p[2])/p[1])) + p[3]


#----------------------------------------------------------------------------#
# Defining a Loss Function to Optimize
#----------------------------------------------------------------------------#

# list for storing the number of optimizer iterations
iterations = []

# list for storing mse of actual and predicted from train at each iteration
loss_train = []

# list fpr storing mse of actual and predicted from val at each iteration
loss_val = []

# initialize the number of iterations as zero
iteration = 0

# loss function for computing mse using actual and predictions from model_func
def loss(p):
    
    # allow modification of these variables outside of the function scope
    global iteration, iterations, loss_train, loss_val
    
    # make predictions using the logistic model with train and current params
    y_hat_train = model_func(x_train, p)
    
    # make predictions using the logistic model with val and current params
    y_hat_val = model_func(x_val, p)
    
    # compute train mse using the actual and predicted age classes from train
    training_loss = sum((y_train - y_hat_train)**2)/len(y_train)
    
    # compute val mse using the actual and predicted age classes from val
    validation_loss = sum((y_val - y_hat_val)**2)/len(y_val)
    
    # store the mse of this current iteration into the training loss list
    loss_train.append(training_loss)
    
    # store the mse of this current iteration into the validation loss list
    loss_val.append(validation_loss)
    
    # store the current iteration of the optimizer
    iterations.append(iteration)
    
    # increment the number of iterations of the optimizer
    iteration += 1
    
    # return result of loss function on train to continue optimizing params
    return training_loss


#----------------------------------------------------------------------------#
# Training the Model Using SciPy Optimizer
#----------------------------------------------------------------------------#

# make an initial guess for the ideal fitting parameters
p0 = [0.76725044, 0.52684187, 0.88343076, 0.67510788]

# search for the optimal params that minimize the output of loss func on train
res = minimize(loss, p0, method = 'Nelder-Mead', tol = 1e-15)

# store the optimal params in a variable
popt = res.x

# print the optimal parameters of the logistic age(weight) model
print('OPTIMAL PARAM:', popt)


#----------------------------------------------------------------------------#
# Unnormalizing
#----------------------------------------------------------------------------#

# save the parameters to variables

A = popt[0]
w = popt[1]
x0 = popt[2]
S = popt[3]

# use the model to make predictions with the train set
y_hat_train = A/(1 + 1/np.exp((x_train - x0)/w)) + S

# use the model to make predictions with the validation set
y_hat_val = A/(1 + 1/np.exp((x_val - x0)/w)) + S

# use the model to make predictions with the entire set
y_hat_all = A/(1 + 1/np.exp((x - x0)/w)) + S

# convert weight values from train and val back to original space
x_train_orig = np.std(customer_df['Weight'])*x_train + np.mean(customer_df['Weight'])
x_val_orig = np.std(customer_df['Weight'])*x_val + np.mean(customer_df['Weight'])

# convert actual values and predicted values from train back to original space
y_train_orig = np.std(customer_df['Age_Class'])*y_train + np.mean(customer_df['Age_Class'])
y_hat_train_orig = np.std(customer_df['Age_Class'])*y_hat_train + np.mean(customer_df['Age_Class'])

# convert actual and predicted values from validation back to original space
y_val_orig = np.std(customer_df['Age_Class'])*y_val + np.mean(customer_df['Age_Class'])
y_hat_val_orig = np.std(customer_df['Age_Class'])*y_hat_val + np.mean(customer_df['Age_Class'])

# convert all predicted values back to original space
y_hat_all_orig = np.std(customer_df['Age_Class'])*y_hat_all + np.mean(customer_df['Age_Class'])


#----------------------------------------------------------------------------#
# Visualizing Model Performance & Results
#----------------------------------------------------------------------------#

# visualize train and val loss as number of optimizer iterations increases

Data().plot_loss(iterations, loss_train, loss_val)

# visualize model results by graphing original data and model fit line

Data().plot_model(x_train_orig,          # unnormalized weight train data
                  y_train_orig,          # unnormalized age class train data
                  x_val_orig,            # unnormalized weight val data
                  y_val_orig,            # unnormalized age class val data
                  customer_df['Weight'], # unnormalized weight data (all)
                  y_hat_all_orig)        # unnormalized age class preds (all)

# form a parity plot to check congruity of model predictions and actual data

Data().plot_parity(y_train_orig,     # unnormalized age class training data
                   y_hat_train_orig, # unnornalized age class preds with train
                   y_val_orig,       # unnormalized age class val data
                   y_hat_val_orig)   # unnormalized age class preds with val

