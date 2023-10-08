
# In[1]:


import pandas as pd
import numpy as np



# In[2]:


data = pd.read_excel('C:/Users/tao24/Desktop/fund_managers.xlsx')
type(data)



# In[3]:


print(data)


# In[4]:


data.head()



# In[5]:


print(data['Risky Ricky'].mean(), data['Safe Steve'].mean())
print(data['Risky Ricky'].std(), data['Safe Steve'].std())
print(data['Risky Ricky'].skew(), data['Safe Steve'].skew())
print(data['Risky Ricky'].kurtosis(), data['Safe Steve'].kurtosis())



# In[6]:


data.describe()



# In[7]:


data.corr()




# In[8]:


cashflow = pd.Series(index=[0,1,2,3,4,5], name='Cashflow',                        data=[-107,5,5,5,5,105])
print(cashflow)

np.irr(cashflow)

