
# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'macro.xls', index_col=0)
data.head()


# In[2]:


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
   
data = pd.DataFrame({'dspread' : data['BMINUSA'] -                                  data['BMINUSA'].shift(1),
                    'dcredit' : data['CCREDIT'] - \
                                data['CCREDIT'].shift(1),
                    'dprod' : data['INDPRO'] - \
                              data['INDPRO'].shift(1),
                    'rmsoft' : LogDiff(data['MICROSOFT']),
                    'rsandp' : LogDiff(data['SANDP']),
                    'dmoney' : data['M1SUPPLY'] - \
                               data['M1SUPPLY'].shift(1),
                    'inflation' : LogDiff(data['CPI']),
                    'term' : data['USTB10Y'] - data['USTB3M'],
                    'dinflation' : LogDiff(data['CPI']) - \
                                   LogDiff(data['CPI']).shift(1),
                    'mustb3m' : data['USTB3M']/12,
                    'rterm' : (data['USTB10Y'] - data['USTB3M']) - \
                              (data['USTB10Y'] - data['USTB3M']).shift(1),
                    'ermsoft' : LogDiff(data['MICROSOFT']) - \
                                data['USTB3M']/12,
                    'ersandp' : LogDiff(data['SANDP']) - \
                                data['USTB3M']/12})
data.head()


# In[3]:


import pickle

with open(abspath + 'macro.pickle', 'wb') as handle:
    pickle.dump(data, handle)

# In[4]:


formula = 'ermsoft ~ ersandp + dprod + dcredit +                 dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit()
print(results.summary())


# In[5]:


hypotheses = 'dprod = dcredit = dmoney = dspread = 0'

f_test = results.f_test(hypotheses)
print(f_test)


