
# In[1]:


import pandas as pd
import numpy as np
import pickle

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD'                     '/QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'capm.xls', index_col=0)

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_sandp' : LogDiff(data['SandP']),
                    'ret_ford' : LogDiff(data['FORD']),
                    'USTB3M' : data['USTB3M']/12,
                    'ersandp' : LogDiff(data['SandP']) - data['USTB3M']/12,
                    'erford' : LogDiff(data['FORD']) - data['USTB3M']/12})

with open(abspath + 'capm.pickle', 'wb') as handle:
    pickle.dump(data, handle)


# In[2]:


with open(abspath + 'capm.pickle', 'rb') as handle:
    data = pickle.load(handle)


# In[3]:


import statsmodels.formula.api as smf
# F-test: multiple hypothesis tests
formula = 'erford ~ ersandp'
hypotheses = 'ersandp = Intercept = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

