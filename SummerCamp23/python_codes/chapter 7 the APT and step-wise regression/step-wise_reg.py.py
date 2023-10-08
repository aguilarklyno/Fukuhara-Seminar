
# In[1]:


import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def forward_selected(data, endog, exg):
    '''
    Linear model designed by forward selection based on p-values.

    Parameters:
    -----------
    data : pandas DataFrame with dependent and independent variables

    endog: string literals, dependent variable from the data
    
    exg: string literals, independent variable from the data
        
    Returns:
    --------
    res : an "optimal" fitted statsmodels linear model instance
           with an intercept selected by forward selection
    '''
    remaining = set(data.columns)
    remaining = [e for e in remaining if (e not in endog)&(e not in exg)]
    exg = [exg]

    scores_with_candidates = []
    for candidate in remaining:
        formula = '{} ~ {}'.format(endog,' + '.join(exg + [candidate]))

        score = smf.ols(formula, data).fit().pvalues[2]
        scores_with_candidates.append((score, candidate))
    scores_with_candidates.sort()

    for pval,candidate in scores_with_candidates:
        if pval < 0.2:    
            exg.append(candidate)

    formula = '{} ~ {}'.format(endog, ' + '.join(exg))
    res = smf.ols(formula, data).fit()
    return res



# In[2]:


import pickle

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns
data.head()


# In[3]:


res = forward_selected(data,'ermsoft','ersandp')

print(res.model.formula)


# In[4]:


print(res.summary())


# In[5]:


print(res.rsquared_adj)

