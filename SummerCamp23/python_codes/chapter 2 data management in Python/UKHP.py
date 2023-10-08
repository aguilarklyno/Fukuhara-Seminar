

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_excel('C:/Users/tao24/Desktop/UKHP.xls', index_col=0)
data.head()


# In[2]:


print(data['Average House Price'].mean())
print(data['Average House Price'].std())
print(data['Average House Price'].skew())
print(data['Average House Price'].kurtosis())

data.describe()


# In[3]:


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    return x_diff


# In[4]:


data['dhp'] = LogDiff(data['Average House Price'])
data.head()


# In[5]:


data.describe()


# In[6]:


data1 = pd.DataFrame({'dhp':LogDiff(data['Average House Price'])})
data1 = data1.dropna()
data1.head()



# In[7]:


import matplotlib.pyplot as plt

plt.figure(1, dpi=600)
plt.plot(data['Average House Price'], label='hp')

plt.xlabel('Date')
plt.ylabel('Average House Price')
plt.title('Graph')
plt.grid(True)

plt.legend()
plt.show()


# In[8]:


plt.figure(2, dpi=600)
plt.hist(data1['dhp'], 20, edgecolor='black', linewidth=1.2)
plt.xlabel('dhp')
plt.ylabel('Density')
plt.title('Histogram')
plt.show()


# In[9]:


data.to_excel('C:/Users/tao24/Desktop/UKHP_workfile.xls')


# In[10]:


import pickle

with open('C:/Users/tao24/Desktop/UKHP.pickle', 'wb') as handle:
    pickle.dump(data, handle)


# In[11]:


with open('C:/Users/tao24/Desktop/UKHP.pickle', 'rb') as handle:
    data = pickle.load(handle)


