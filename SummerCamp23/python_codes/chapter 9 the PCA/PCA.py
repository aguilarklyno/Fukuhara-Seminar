
# In[1]:


from numpy import mean,cov,cumsum,dot,linalg,std,sort
import pandas as pd
import matplotlib.pyplot as plt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
    
data = pd.read_excel(abspath + 'FRED.xls', index_col=0)

data.head()



# In[2]:


def princomp(data):    
    '''
    Computing eigenvalues and eigenvectors of covariance matrix
    
    Parameters: 
    -----------
    data: the m-by-n DataFrame which corresponds m rows of observations 
          and n columns of variables.
    
    Returns:
    -----------
    coeff: a n-by-n matrix (eigenvectors) where it contains all coefficients 
           of principal component for each variable.

    latent: a vector of the eigenvalues 
    '''
    
    M = data.apply(lambda x: (x-mean(x))/std(x) ) # normalize
    
    # Note: cov inputs require a column vector;
    # That is each row of m represents a variable,
    # and each column is a single observation of all those variables
    # To tackle this, simply add argument rowvar=True
    # or transpose M matrix
    [latent,coeff] = linalg.eig(cov(M.transpose()))
    # attention: latent (eigenvalues) is not always sorted
    latent = sort(latent)[::-1]
    
    # convert arrays to DataFrame or Series
    coeff = pd.DataFrame(coeff.T, columns=data.columns)
    latent = pd.Series(latent, name='Eigenvalue')
    
    return coeff, latent


# In[3]:


coeff, latent = princomp(data)


# In[4]:


coeff


# In[5]:


perc_lat = cumsum(latent)/sum(latent)
print(perc_lat)


# In[6]:


plt.figure(1, dpi=600)
plt.stem(range(len(perc_lat)),perc_lat,'--b')
plt.ylabel('Percentage of Eeigenvalues')
plt.xlabel('The number of components')
plt.show()

