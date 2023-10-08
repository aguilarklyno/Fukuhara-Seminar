
# In[1]:


import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pickle

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns


# In[2]:


# regression
formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit()
y_fitted = results.fittedvalues
residuals = results.resid



# In[3]:


plt.figure(1,dpi=600)
plt.plot(residuals, label='resid')
plt.plot(y_fitted, label='linear prediction')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid(True)
plt.legend()
plt.show()


# In[4]:


residuals.nsmallest(2)


# In[5]:


data['APR00DUM'] = np.where(data.index == '2000-4-1', 1, 0)
data['DEC00DUM'] = np.where(data.index == '2000-12-1', 1, 0)

# regression
formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney + dspread + rterm +                      APR00DUM + DEC00DUM'
results = smf.ols(formula, data).fit()
print(results.summary())

