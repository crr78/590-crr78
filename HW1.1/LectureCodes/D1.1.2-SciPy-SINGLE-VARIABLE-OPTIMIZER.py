# #--------------------------------
# #OPTIMZE A FUNCTION USING SCIPY
# #--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# "minimize" used for minimization of scalar function of one or more variables
from scipy.optimize import minimize


# ax.plot(xe, ye, '-', label='Ground-Truth')
# ax.plot(xe, model(xe, popt), 'r-', label="Model")


#FUNCTION TO OPTIMZE
def f(x):
	out=x**2.0
	# out=(x+10*np.sin(x))**2.0
	return out

#PLOT
#DEFINE X DATA FOR PLOTTING
N=1000; xmin=-20; xmax=20
X = np.linspace(xmin,xmax,N)

plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('x', fontsize=FS)
plt.ylabel('f(x)', fontsize=FS)
plt.plot(X,f(X),'-')

num_func_eval=0
def f1(x):
    # allows modification of the variable outside of the scope of the function
	global num_func_eval
	out=f(x)
	num_func_eval+=1
    # every 10 times the function is evaluated, 
	if(num_func_eval%10==0):
		print(num_func_eval,x,out)
	plt.plot(x,f(x),'ro')
	plt.pause(0.11)

	return out

#INITIAL GUESS 
xo=xmax #
#xo=np.random.uniform(xmin,xmax)
print("INITIAL GUESS: xo=",xo, " f(xo)=",f(xo))
# f1 is the objective function to be minimized
# method is the type of solver to use. In this case, "Nelder-Mead" alg is used
# tol sets the tolerance for termination. Specifying this causes the selected
# minimizaiton algorithm to set some relevant solver-specific tolerance(s)
# equal to tol

# minimize() in this case appears to search for the optimal parameter for x 
# that results in the minimum value of the function
res = minimize(f1, xo, method='Nelder-Mead', tol=1e-5)

# returning the optimal parameter
popt=res.x
print("OPTIMAL PARAM:",popt)

plt.show()

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND IT COMPLETELY


