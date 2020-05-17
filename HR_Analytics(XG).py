#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # DATA PREPROCESSING

# In[2]:


data = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/HR_analytics/train.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.dtypes


# In[6]:


data.shape


# In[7]:


data.isna().sum()


# In[8]:


data['education']


# In[9]:


data['education'][data['education'].isna()] = data['education'].mode()[0]


# In[10]:


data.isna().sum()


# In[11]:


data['previous_year_rating'][data['previous_year_rating'].isna()] = data['previous_year_rating'].mean()


# In[12]:


data.isna().sum()


# In[13]:


databackup = data.copy()


# In[14]:


data.shape


# In[15]:


data.dtypes


# In[16]:


data.head()


# In[17]:


data.corr()


# In[18]:


corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))
g = sns.heatmap(data[top_corr_features].corr(), annot = True)


# In[19]:


y = data['is_promoted']
x = data.drop(['employee_id', 'is_promoted'] , axis = 1)


# In[20]:


x.head()


# In[21]:


x = pd.get_dummies(x)


# In[22]:


x.shape


# In[23]:


x.head()


# # XGBOOST

# In[24]:


parameters = {
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[25]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[26]:


classifier=xgboost.XGBClassifier()


# In[27]:


random_search=RandomizedSearchCV(classifier, parameters, n_iter=5, scoring='f1', n_jobs=-1, cv=10, verbose=3)


# In[28]:


random_search.fit(x, y)


# In[29]:


x.head()


# In[49]:


random_search.best_estimator_.predict(test_x)


# In[31]:


random_search.best_params_


# In[32]:


classifier = xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.2, max_delta_step=0, max_depth=8,
              min_child_weight=3, missing=None, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)


# In[33]:


from sklearn.model_selection import cross_val_score
f1_scores = cross_val_score(classifier, x, y, cv=10, scoring='f1')


# In[34]:


f1_scores


# In[35]:


f1_scores.mean()


# # TEST DATA

# In[36]:


test = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/HR_analytics/test.csv")


# In[37]:


test.head()


# In[38]:


test.isna().sum()


# In[39]:


test['education'][test['education'].isna()] = test['education'].mode()[0]


# In[40]:


test['previous_year_rating'][test['previous_year_rating'].isna()] = test['previous_year_rating'].mean()


# In[41]:


test.isna().sum()


# In[42]:


test_x = test.drop('employee_id', axis = 1)


# In[43]:


test_x = pd.get_dummies(test_x)


# In[44]:


classifier.fit(x, y)


# In[45]:


y_prediction = classifier.predict(test_x)


# In[46]:


print(y_prediction)


# In[47]:


my_submission = pd.DataFrame({'employee_id': test.employee_id, 'is_promoted': y_prediction})


# In[48]:


my_submission.to_csv('submission_XG2.csv', index = False)


# In[ ]:




