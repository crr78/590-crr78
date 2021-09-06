

"""
Chevy Robertson (crr78@georgetown.edu)
Neural Nets & Deep Learning
HW1.1, Part-III: Regression Workflow (Visual EDA Script)
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


#----------------------------------------------------------------------------#
# "Data" class for reading, partitioning, and visualizing the data
#----------------------------------------------------------------------------#

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
        
    # method for generating a box plot to observe weight dist. by age class
    def make_box(self):
        
        # for plotting purposes, make sure 'Age_Class' is categorical
        self.df['Age_Class'] = self.df['Age_Class'].astype('category')
        
        # forming a box plot of weight distribution by age classification
        
        self.df.boxplot('Weight', 
                        by = 'Age_Class', 
                        patch_artist = True, # adds color to the boxplot
                        showmeans = True)    # plots the mean of each group
        
        # remove the title that comes with the plot
        plt.suptitle('')
        
        # add a title
        plt.title('Weight Distribution for Children vs. Adults')
        
        # add an x-label
        plt.xlabel('Age Classification')
        
        # add a y-label
        plt.ylabel(self.ylab)
        
        # add xticks to indicate the groups
        plt.xticks(ticks = [1, 2], labels = ['Children', 'Adults'])
        
        # show the plot
        plt.show()
        
        return None
    
    # method for generating a pie chart to observe age group proportion
    def make_pie(self):
        
        # group the dataframe by age group and count the customers
        age_counts = self.df.groupby('Age_Class').count()
        
        # determine the proportions
        props = age_counts['Weight']/len(self.age_class)*100
        
        # for plotting purposes redefine the index of the aggregated df
        props.index = ['Children', 'Adults']
        
        # form a dataframe from the proportions for plotting purposes
        props_df = pd.DataFrame(props)

        # establish a figure object and axes object for plotting purposes
        fig, ax = plt.subplots()
        
        # make a pie chart using the proportions and label accordingly
        
        ax.pie(props_df['Weight'],      # pie chart sizes 
               labels = props_df.index, # labels are the age groups
               explode = (0.1, 0),      # make the children slice stand apart
               autopct = '%1.1f%%',     # format percentages
               shadow = True,           # cast a shadow on the pie chart
               startangle = 90)         # pie chart starts at 90˚
        
        # make sure that the pie chart is drawn as a circle
        ax.axis('equal')
        
        # add a title
        plt.title('Proportion of Customers by Age Classification')
        
        # show the plot
        plt.show()
        
        return None
    
    # method for generating a scatterplot to show relationship between weight
    # and age for the customers
    def make_scatter(self):
        
        # make a variable to store the ages of the children
        x_c = np.array(self.df[self.df['Age_Class'] == 0]['Age'])
        
        # make another variable to store the weights of the children
        y_c = np.array(self.df[self.df['Age_Class'] == 0]['Weight'])
        
        # make a variable to store the ages of the adults
        x_a = np.array(self.df[self.df['Age_Class'] == 1]['Age'])
        
        # make another variable to store the weights of the adults
        y_a = np.array(self.df[self.df['Age_Class'] == 1]['Weight'])
        
        # plot the age and weight data for the children
        plt.plot(x_c, y_c, 'r.', label = 'Child')
        
        # plot the age and weight data for the adults
        plt.plot(x_a, y_a, 'k.', label = 'Adult')
        
        # add a title
        plt.title('Relationship Between Weight and Age for the Customers')
        
        # add an x-label
        plt.xlabel(self.xlab)
        
        # add a y-label
        plt.ylabel(self.ylab)
        
        # add a grid
        plt.grid()
        
        # add a legend
        plt.legend()
        
        # show the plot
        plt.show()
        
        return None


#----------------------------------------------------------------------------#
# "Main" Function
#----------------------------------------------------------------------------#
        
# "main" function for calling on functions and class methods    
def main():
    
    # initialize an object of class Data
    data_obj = Data()
    
    # use make_box() method for EDA using a box plot
    data_obj.make_box()
    
    # use make_pie() method for EDA using a pie chart
    data_obj.make_pie()
    
    # use make_scatter() method for EDA using a scatterplot
    data_obj.make_scatter()

# if the python interpreter is running this source file as the main program    
if __name__ == '__main__':
    
    # call main() function to execute program functions and class Data methods
    main()


#----------------------------------------------------------------------------#
# Sources
#----------------------------------------------------------------------------#

# Inspiration for Pie Chart:
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html


