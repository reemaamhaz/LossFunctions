
# 1. Define loss functions and find their minimum values.
# 2. Explore different techniques for finding the maximum or mininum of a function.

import pandas as pd
import numpy as np
np.random.seed(42)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.optimize import minimize



import dsua_112_utils
import sys, os, pickle
home_path = os.environ["HOME"]


def squared_loss(y_obs, c):
    return (y_obs - c)**2


y_obs = 10
c_values = np.linspace(0, 20, 100) # some values of c


plt.plot(c_values, squared_loss(y_obs, c_values))

    
plt.xlabel('c')
plt.ylabel('L2 loss')


df = sns.load_dataset("tips")
tips = np.array(df['tip']) # array of observed tips




def mean_squared_error(c, data):
    loss = []
    for tips in data:
        loss.append(squared_loss(tips, c))
    return (1/len(data)*(sum(loss)))


c_values = np.linspace(0, 6, 100)

plt.plot(c_values, mean_squared_error(c_values, tips))

plt.xlabel('c')
plt.ylabel('L2 loss')


min_observed_mse = 3
min_observed_mse


x_values = np.linspace(-4, 2.5, 100)

def fx(x):
    return 0.1 * x**4 + 0.2*x**3 + 0.2 * x **2 + 1 * x + 10

plt.plot(x_values, fx(x_values))

minimize(fx, x0 = 1.1)


minimization_result_for_fx = minimize(fx, x0 = 0)
min_of_fx = minimization_result_for_fx['fun']
x_which_minimizes_fx = minimization_result_for_fx['x'][0]



w_values = np.linspace(-2, 10, 100)

def fw(w):
    return 0.1 * w**4 - 1.5*w**3 + 6 * w **2 - 1 * w + 10

plt.plot(w_values, fw(w_values))



minimize(fw, x0 = 6.5)


def mean_squared_error_with_hard_coded_data(c):
    return mean_squared_error(c, tips)
min_scipy = minimize(mean_squared_error_with_hard_coded_data, x0=0.0)['x'][0]


min_computed = tips.mean()
min_computed


def abs_loss(c, y_obs):
    return abs(y_obs - c)


y_obs = 10
c_values = np.linspace(0, 20, 100) # some arbitrary values of c

plt.plot(c_values, abs_loss(c_values,y_obs))
plt.xlabel('c')
plt.ylabel('L1 loss')

q4b_gca = plt.gca(); 


def mean_absolute_error(c, data):
    loss = []
    for tips in data:
        loss.append(abs_loss(c, tips))
    return (1/len(data)*(sum(loss)))

c_values = np.linspace(0, 6, 100)

plt.plot(c_values, mean_absolute_error(c_values, tips))
plt.xlabel('c')
plt.ylabel('L1 loss')



c_values = np.linspace(2.7, 3.02, 100)

plt.plot(c_values, mean_absolute_error(c_values, tips))
plt.xlabel('c')
plt.ylabel('L1 loss')



min_observed_mae = 2.9




def mean_absolute_error_of_data(c):
     return mean_absolute_error(c, tips)
min_abs_scipy = minimize(mean_absolute_error_of_data, x0=0)['x'][0]







min_abs_computed = tips.mean()



