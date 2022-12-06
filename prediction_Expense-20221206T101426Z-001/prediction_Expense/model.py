#!/usr/bin/env python
# coding: utf-8

# In[39]:


####################Setting working directory############################
import os
os.chdir("F:\Model Deployment\prediction_Expense")


# In[40]:


################################Step 1 - Import data############################################
import pandas as pd
MyData = pd.read_csv("Income_Expense_Data.csv")


# In[41]:


#Checking Size of data
MyData.shape


# In[42]:


#Checking first few records
MyData.head(10)


# In[43]:


################Step 2-Data Cleaning#######################
#Check for missing values
MyData.isnull().sum() 


# In[44]:


#Treating null value-replacing null value with median
MyData["Income"].fillna((MyData["Income"].median()), inplace = True)


# In[45]:


#Check for missing values - Again
MyData.isnull().sum() 


# In[46]:


#Checking for outliers
MyData.describe()  #notice the maximum value in Age


# In[47]:


#Checking different percentiles
pd.DataFrame(MyData['Age']).describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))


# In[48]:


#checking boxplot for Age column
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(MyData['Age'])
plt.show()


# In[49]:


#Checking Outlier by definition and treating outliers

#getting median Age
Age_col_df = pd.DataFrame(MyData['Age'])
Age_median = Age_col_df.median()

#getting IQR of Age column
Q3 = Age_col_df.quantile(q=0.75)
Q1 = Age_col_df.quantile(q=0.25)
IQR = Q3-Q1

#Deriving boundaries of Outliers
IQR_LL = int(Q1 - 1.5*IQR)
IQR_UL = int(Q3 + 1.5*IQR)

#Finding and treating outliers - both lower and upper end
MyData.loc[MyData['Age']>IQR_UL , 'Age'] = int(Age_col_df.quantile(q=0.99))
MyData.loc[MyData['Age']<IQR_LL , 'Age'] = int(Age_col_df.quantile(q=0.01))


# In[50]:


#Check max age value now
max(MyData['Age'])


# In[35]:


################Step 3-Exploratory data analysis#######################
#Check how Expense is varying with income
x = MyData["Income"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Expense")


# In[51]:


#Check how Expense is varying with Age
x = MyData["Age"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Age")


# In[52]:


#check correltion matrix - to check the strength of variation bwtween two variables
correlation_matrix= MyData.corr().round(2)
f, ax = plt.subplots(figsize =(8, 4)) 
import seaborn as sns
sns.heatmap(data=correlation_matrix, annot=True)


# In[53]:


################Step 4-feature engineering#######################
#Normalization/scaling of data - understanding scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(MyData)
scaled_data


# In[55]:


#converting data back to pandas dataframe
MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ["Age","Income","Expense"]


# In[56]:


#Separating features and response
features = ["Income","Age"]
response = ["Expense"]
X=MyData_scaled[features]
y=MyData_scaled[response]


# In[57]:


#Dividing data in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Importing neccesary packages
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[58]:


#Fitting lineaar regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[59]:


#Checking accuracy on test data
accuracy = model.score(X_test,y_test)
print(accuracy*100,'%')


# In[60]:


model.predict(X_test) #predcited values on test data


# In[61]:

#Dumping the model object
import pickle
pickle.dump(model, open('model.pkl','wb'))


#Reloading the model object
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30000, 24]]))


# In[ ]:




