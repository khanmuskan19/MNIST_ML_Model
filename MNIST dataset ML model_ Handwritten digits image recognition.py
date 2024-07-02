#!/usr/bin/env python
# coding: utf-8

# # Fetching Data

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist= fetch_openml('mnist_784', version=1)


# In[3]:


# mnist 
# the matrix of 28x28 is basically flattend to 784 as a row here!You can convert a row vector back into a 28x28 matrix using .reshape(28, 28) to visualize the image correctly.
# You can convert a row vector back into a 28x28 matrix using .reshape(28, 28) to visualize the image correctly.


# In[4]:


x=mnist['data']
y=mnist['target']


# In[5]:


x


# In[6]:


y


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib
import matplotlib.pyplot as plt


# In[9]:


x = mnist.data.to_numpy()
some_digit=x[40000]
some_digit_img=some_digit.reshape(28,28)

# plt.imshow(some_digit_img, cmap=matplotlib.cm.binary, interpolation="nearest")


# In[10]:


plt.imshow(some_digit_img, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')


# In[11]:


import numpy as np
shuffle_index=np.random.permutation(60000)
x_train, ytrain=x_train[shuffle_index],y_train[shuffle_index]


# In[ ]:


x_train, x_test=x[:60000],x[60000:]
y_train, y_test=y[:60000],y[60000:]


# In[ ]:


shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index], y_train.[shuffle_index]


# In[ ]:


# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_7 = (y_train == 7)
y_test_7 = (y_test == 7)


# In[ ]:


y_test_2


# In[ ]:


# Train a logistic regression classifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(tol=0.1, solver='lbfgs')
clf.fit(x_train, y_train_2)


# In[ ]:


clf.predict([some_digit])


# In[ ]:


# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")


# In[ ]:


a.mean()


# In[ ]:




