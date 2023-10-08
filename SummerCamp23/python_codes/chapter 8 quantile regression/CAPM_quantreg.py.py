# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import matplotlib.pyplot as plt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'

with open(abspath + 'capm.pickle', 'rb') as handle:
    data = pickle.load(handle)
data = data.dropna()


# In[2]:


# regression
# quantile(50)
res = smf.quantreg('erford ~ ersandp', data).fit(q=0.5)
print(res.summary())



# In[3]:


# Simultaneous-quantile regression
# 10 20 30 40 50 60 70 80 90, ten quantiles
quantiles = np.arange(0.10, 1.00, 0.10)

for x in quantiles:
    print('-----------------------------------------------')
    print('{0:0.01f} quantile'.format(x))
    res = smf.quantreg('erford ~ ersandp', data).fit(q=x)
    print(res.summary())


# In[4]:


def model_paras(data, quantiles):
    '''
    Parameters: 
    -----------
        data: pandas DataFrame with dependent and independent variables
        quantiles: quantile number
        
    Returns:
    -----------
        quantreg_res: pandas DataFrame with model parameters for each 
                      quantile regression specification
        y_hat: pandas DataFrame with all the fitted value of y
    '''
    parameters = []
    y_pred = {}
    for q in quantiles:
        res = smf.quantreg('erford ~ ersandp', data).fit(q=q)  
        # obtain regression's parameters
        alpha = res.params['Intercept']
        beta = res.params['ersandp']
        lb_pval = res.conf_int().loc['ersandp'][0]
        ub_pval = res.conf_int().loc['ersandp'][1]
        # obtain the fitted value of y
        y_pred[q] = res.fittedvalues
        # save results to lists
        parameters.append((q,alpha,beta,lb_pval,ub_pval))
       
    quantreg_res = pd.DataFrame(parameters, columns=['q', 'alpha',                                                      'beta','lb','ub'])
    y_hat = pd.DataFrame(y_pred)  
    return quantreg_res, y_hat


# In[5]:


quantreg_paras, y_hats = model_paras(data, quantiles)


# In[6]:


print(quantreg_paras)


# In[7]:


y_hats.head()


# In[8]:


plt.figure(1, dpi=600)
for i in quantreg_paras.q:
    x = data['ersandp']
    y = y_hats[i]
    if i == 0.50:
        plt.plot(x,y,color='red')        
    else:
        plt.plot(x,y,color='grey')
        

plt.ylabel('ersandp')
plt.xlabel('erford')
plt.show()

