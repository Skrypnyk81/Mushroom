
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[11]:


data = pd.read_csv('mushrooms.csv')


# In[12]:


data.head()


# In[13]:


data = pd.get_dummies(data)


# In[18]:


data = data.drop('class_p', axis=1)


# In[20]:


X = data.drop('class_e', axis=1)
y = data['class_e']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[34]:


lr = LinearRegression().fit(X_train, y_train)


# In[35]:


print('Правильность на обучающем наборе: {:.2f}'.format(lr.score(X_train, y_train)))
print('Правильность на обучающем наборе: {:.2f}'.format(lr.score(X_test, y_test)))
print('Значения правильности перекрестной проверки: {:.2f}'.format(cross_val_score(lr, X_train, y_train, cv=5).mean()))


# In[38]:


clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)


# In[39]:


print('Правильность на обучающем наборе: {:.2f}'.format(clf.score(X_train, y_train)))
print('Правильность на обучающем наборе: {:.2f}'.format(clf.score(X_test, y_test)))
print('Значения правильности перекрестной проверки: {:.2f}'.format(cross_val_score(clf, X_train, y_train, cv=5).mean()))

