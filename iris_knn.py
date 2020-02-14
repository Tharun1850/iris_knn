#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[33]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import   KNeighborsClassifier


# In[4]:


iris=load_iris()
print(iris)


# In[15]:


print(iris.data.T)


# In[19]:


details =iris.data.T
print(details[0])
print("---------------------------------------------------------")
print(details[1])
print("---------------------------------------------------------")
print(details[2])
print("---------------------------------------------------------")
print(details[3])


# In[6]:


print(iris.keys())


# In[23]:


print(iris.feature_names)
labels=iris.feature_names


# In[10]:


print(iris.DESCR)


# In[26]:


plt.scatter(details[0],details[1], c=iris.target)
plt.xlabel(labels[0])    


# In[30]:


plt.scatter(details[0],details[1], c=iris.target)
plt.xlabel(labels[0])
plt.ylabel=(labels[1])
plt.show()


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(iris['data'],iris['target'])
knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(x_train,y_train)
x_new =np.array([[5, 2.9 , 1, 0.2]])
prediction = knn.predict(x_new)
print(prediction)


# In[51]:


knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(x_train,y_train)
x_new =np.array([[7,7,7,7]])
prediction = knn.predict(x_new)
print(prediction)


# In[53]:


knn = KNeighborsClassifier(n_neighbors =5)
knn.fit(x_train,y_train)
x_new =np.array([[7,0,0,7]])
prediction = knn.predict(x_new)
print(prediction)


# In[54]:


knn.score(x_train,y_train)


# In[ ]:




