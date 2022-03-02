#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
house = pd.read_csv('C://Users//MY PC//desktop//DATASETS//ml_house_data_set.csv')


# In[2]:


house.head()


# In[3]:


house.info()


# In[4]:


house['unit_number']


# In[5]:


house.info()


# In[6]:


house.columns


# In[7]:


cols = ['zip_code','city','unit_number','street_name','house_number']
house = house.drop(cols, axis=1)


# In[8]:


house.columns


# In[9]:


house.info()


# In[10]:


house['garage_type'].unique()


# In[11]:


garage_dummies = pd.get_dummies(house['garage_type'], prefix = 'garage_type')
house = pd.concat([house, garage_dummies],axis=1)


# In[12]:


house.info()


# In[13]:


house.drop('garage_type',axis=1, inplace=True)


# In[14]:


house.info()


# In[15]:


house.columns


# In[16]:


cols = ['year_built', 'stories', 'num_bedrooms', 'full_bathrooms',
       'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft',
       'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating',
       'has_central_cooling','garage_type_attached',
       'garage_type_detached', 'garage_type_none','sale_price']


# In[17]:


house[cols].corr()


# # Single linear Regression

# In[19]:


train = house[0:int((house.shape[0]*(70/100)))]
test = house[int(house.shape[0]*(70/100)):]


# In[28]:


columns = ['year_built', 'stories', 'num_bedrooms', 'full_bathrooms',
       'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft',
       'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating',
       'has_central_cooling','garage_type_attached',
       'garage_type_detached', 'garage_type_none']


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()

# Training the model
lr.fit(train[['total_sqft']], train['sale_price'])
prediction = lr.predict(train[['total_sqft']])
train_mse = mean_squared_error(prediction, train['sale_price'])
import numpy as np
train_rmse = np.sqrt(train_mse)

# Using the model on the test data
lr.fit(test[['total_sqft']], test['sale_price'])
test_prediction = lr.predict(test[['total_sqft']])
test_mse = mean_squared_error(test_prediction, test['sale_price'])

test_rmse = np.sqrt(test_mse)


# In[27]:


test_rmse


# In[25]:


train_mse


# In[21]:


train_rmse


# In[ ]:


house.shape[0]*(70/100)


# In[ ]:


house.shape[0]


# In[ ]:





# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()

# Training the model
lr.fit(train[columns], train['sale_price'])
prediction = lr.predict(train[columns])
train_mse = mean_squared_error(prediction, train['sale_price'])
import numpy as np
train_rmse = np.sqrt(train_mse)

# Using the model on the test data
lr.fit(test[columns], test['sale_price'])
test_prediction = lr.predict(test[columns])
test_mse = mean_squared_error(test_prediction, test['sale_price'])

test_rmse = np.sqrt(test_mse)


# In[30]:


test_rmse


# In[31]:


lr.intercept_


# In[32]:


lr.coef_


# In[ ]:




