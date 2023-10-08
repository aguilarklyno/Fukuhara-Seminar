
# In[1]:


import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import pickle
import pandas as pd
import matplotlib.pyplot as plt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns


# In[2]:


def get_rss(data):
    '''
    inputs:
        data: a pandas DataFrame of independent and dependent variable
    outputs:
        rss: the sum of residuals
        N: the observations of inputs
        K: total number of parameters
    '''    
    formula = 'ermsoft ~ ersandp + dprod + dcredit +                          dinflation + dmoney + dspread + rterm'
    results = smf.ols(formula, data).fit()
    rss = (results.resid**2).sum() # obtain the residuals sum of square
    N = results.nobs
    K = results.df_model
    return rss, N, K


# In[3]:


# split samples
data1 = data[:'1996-01-01']
data2 = data['1996-01-01':]

# get rss of whole sample
RSS_total, N_total, K_total = get_rss(data)
# get rss of the first part of sample
RSS_1, N_1, K_1 = get_rss(data1)
# get rss of the second part of sample
RSS_2, N_2, K_2 = get_rss(data2)

nominator = (RSS_total - (RSS_1 + RSS_2)) / K_total
denominator = (RSS_1 + RSS_2) / (N_1 + N_2 - 2*K_total)

result = nominator/denominator


# In[4]:


result


# In[5]:


formula = 'ermsoft ~ ersandp + dprod + dcredit +                      dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit()

name = ['test statistic', 'pval', 'crit']
test = sms.breaks_cusumolsresid(olsresidual = results.resid,                                ddof = results.df_model)
lzip(name, test)


# In[6]:


def recursive_reg(variable, i, interval):
    '''
    Parameters: 
    -----------
        variable: the string literals of a variable name in regression
                  formula.
        i: the serial number of regression.
        interval: the number of consective data points in initial sample
        
    Returns:
    -----------
        coeff: the coefficient estimation of the variable
        se: the standard errors of the variable
    '''
    formula = 'ermsoft ~ ersandp + dprod + dcredit +                      dinflation + dmoney + dspread + rterm'
    results = smf.ols(formula, data.iloc[:i+interval]).fit()
    coeff = results.params[variable]
    se = results.bse[variable]

    return coeff, se


# In[7]:


parameters = []
for i in range(373):
    coeff, se = recursive_reg('ersandp', i, 11)
    
    parameters.append((coeff,se))
    
parameters = pd.DataFrame(parameters, columns=['coeff','se'],                          index = data[:-10].index)

parameters['ersandp + 2*se'] = parameters['coeff'] + 2*parameters['se']
parameters['ersandp - 2*se'] = parameters['coeff'] - 2*parameters['se']



# In[8]:


plt.figure(1, dpi=600)
plt.plot(parameters['coeff'], label=r'$\beta_{ersandp}$')
plt.plot(parameters['ersandp + 2*se'], label=r'$\beta_{ersandp} + 2*SE$',         linestyle='dashed')
plt.plot(parameters['ersandp - 2*se'], label=r'$\beta_{ersandp} - 2*SE$',         linestyle='dashed')

plt.xlabel('Date')
plt.grid(True)
plt.legend()
plt.show()

