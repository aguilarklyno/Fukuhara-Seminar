
# In[1]:


import statsmodels.formula.api as smf
import pickle

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
     
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns



# In[2]:


formula = 'ermsoft ~ ersandp + dprod + dcredit +                      dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit(cov_type='HC1')
print(results.summary())


