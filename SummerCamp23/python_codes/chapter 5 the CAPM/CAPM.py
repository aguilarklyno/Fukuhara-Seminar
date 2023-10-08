

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD'                 '/QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'capm.xls', index_col=0)

data.head()



# In[2]:


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_sandp' : LogDiff(data['SandP']),
                    'ret_ford' : LogDiff(data['FORD']),
                    'USTB3M' : data['USTB3M']/12,
                    'ersandp' : LogDiff(data['SandP']) - data['USTB3M']/12,
                    'erford' : LogDiff(data['FORD']) - data['USTB3M']/12})
data.head()



# In[3]:


plt.figure(1, dpi=600)
plt.plot(data['ersandp'], label='ersandp')
plt.plot(data['erford'], label='erford')

plt.xlabel('Date')
plt.ylabel('ersandp/erford')
plt.title('Graph')
plt.grid(True)

plt.legend()
plt.show()



# In[4]:


plt.figure(2, dpi=600)

plt.scatter(data['ersandp'], data['erford'])

plt.xlabel('ersandp')
plt.ylabel('erford')
plt.title('Graph')
plt.grid(True)

plt.show()


# In[5]:


formula = 'erford ~ ersandp'
results = smf.ols(formula, data).fit()
print(results.summary())



# In[6]:


# F-test: hypothesis testing
formula = 'erford ~ ersandp'
hypotheses = 'ersandp = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

