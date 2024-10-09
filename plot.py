#%%

import matplotlib.pyplot as plt
import numpy as np

#  X - independent features(excluding target variable)
# y - dependent variables, called (target).

# settimg data for plotting
x = [1,2,3,4,5]
y = [7,3,5,7,11]

# creating a plot 
plt.plot(x,y)
#  a title for the plot
plt.title("My plot") 
# labels for the plot:
plt.xlabel("I.V")
plt.ylabel("Target")

# show plot:
plt.show()
# %%
# 
