#!/usr/bin/env python
# coding: utf-8

# ### import  Libraries ##

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ### Load The Data ###

# In[10]:


df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('fulfilment_center_info.csv')
df3 = pd.read_csv('meal_info.csv')


# In[11]:


combined_df = pd.merge(df1,df3,on='meal_id')


# In[12]:


df = pd.merge(combined_df,df2,on='center_id')


# ### Data Exploring ###

# In[14]:


df


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


df.duplicated().sum()


# In[18]:


df.isna().sum()


# ### Visualisation of the data ###

# In[20]:


for col in df:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('count')
    plt.show()


# In[21]:


num_df = df.select_dtypes(include="number")


# In[22]:


### correlation metrix ###

plt.figure(figsize=(10, 6))

sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# ### Outlier Remover ###

# In[24]:


plt.boxplot(num_df)
plt.show()


# In[25]:


# One hot encoding for columns in training dataset
df = pd.get_dummies(df,columns=['category','cuisine','center_type'],dtype= int,drop_first=True)
df


# In[26]:


x = df.drop('num_orders',axis=1)
y = ['num_orders']


# In[27]:


# min max scaling for faetures having non-gaussian distribution in training dataset
min_scaler = MinMaxScaler()
numerical_colms1 = ['emailer_for_promotion','homepage_featured']
df[numerical_colms1] = min_scaler.fit_transform(df[numerical_colms1])
df


# In[28]:


# standard scaling for features having gaussian distribution in training dataset
std_scaler = StandardScaler()
numerical_colms3 = ['week','center_id','meal_id','city_code','region_code','op_area']
df[numerical_colms3] = std_scaler.fit_transform(df[numerical_colms3])
df


# In[29]:


# Seperating Features and labels
x = df.drop('num_orders',axis=1)
y = df['num_orders']


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=2)


# In[31]:


reg_lin = LinearRegression()
reg_lin.fit(x_train,y_train)
y_pred = reg_lin.predict(x_test)


# In[59]:


# Making Predictions  
y_pred = model.predict(x_test)    

# Calculate evaluation metrics  
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

# Print the evaluation metrics  
print(f'Mean Absolute Error (MAE): {mae:.2f}')  
print(f'Mean Squared Error (MSE): {mse:.2f}')  
print(f'R² Score: {r2:.2f}')


# ### Polynomial Regression

# In[ ]:


poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_reg = LinearRegression()
poly_reg.fit(x_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f'Mean Squared Error (MSE): {mse_poly:}')


# ## Lasso Regression

# In[34]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = Lasso(alpha=0.1)
model.fit(x, y)
y_pred = model.predict(x_test)
mse_poly = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse_poly:}') 


# In[61]:


result_df = pd.DataFrame(y_pred)
result_df


# In[63]:


id_series = pd.Series(id)
result_df = pd.concat([id_series,result_df],axis = 1)
result_df.rename(columns={0 : 'num_orders'}, inplace=True)


# In[65]:


result_df = result_df.iloc[:,-1:2]


# In[67]:


result_df


# In[ ]:




