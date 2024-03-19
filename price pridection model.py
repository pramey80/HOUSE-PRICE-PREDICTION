#!/usr/bin/env python
# coding: utf-8

# ## "Predicting House Prices"
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[2]:


df = pd.read_csv("housing.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df['CHAS'].value_counts()


# In[6]:


df.describe()


# In[7]:


df.hist(bins=50, figsize=(20 ,15))


# ## Train-test spliting
# 

# In[8]:


def split_train_test(data, test_ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


train_set, test_set = split_train_test(df, 0.2)


# In[10]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['CHAS']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[13]:


strat_test_set


# In[14]:


strat_test_set.describe()


# strat_test_set.info()

# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


df = strat_train_set.copy()


# ## Correlation
# 

# In[18]:


corr_matrix = df.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


attributes = ["MEDV","RM" , "ZN" , "LSTAT"]
scatter_matrix(df[attributes] , figsize = (12,8))


# In[21]:


df.plot(kind="scatter" ,x="RM" ,y="MEDV" , alpha=0.8)


# ## ATTRIBUTES COMBINATION
# 

# In[22]:


df["TAXRM"] = df['TAX']/df['RM']


# In[23]:


df.head()


# In[24]:


corr_matrix = df.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[25]:


df.plot(kind="scatter" , x="TAXRM" , y="MEDV" , alpha=0.8)


# In[26]:


df = strat_train_set.drop("MEDV",axis=1)
df_labels = strat_train_set["MEDV"].copy()


# ## MA

# In[27]:


df.dropna(subset=["RM"])
df.shape


# In[28]:


df.drop("RM" , axis=1)


# In[29]:


df.drop("RM" , axis=1).shape


# In[30]:


median = df["RM"].median()


# In[31]:


median


# In[32]:


df["RM"].fillna(median)


# In[33]:


df.shape


# In[34]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(df)


# In[35]:


imputer.statistics_


# In[36]:


X = imputer.transform(df)


# In[37]:


df_tr =pd.DataFrame(X, columns=df.columns)


# In[38]:


df_tr.describe()


# In[39]:


df.shape


# In[41]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scale", StandardScaler()),
])


# In[42]:


df_num_tr=my_pipeline.fit_transform(df)


# In[43]:


df_num_tr.shape


# ## Selecting a desired model
# 

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
model = DecisionTreeRegressor()
model.fit(df_num_tr, df_labels)


# In[45]:


some_data = df.iloc[:5]


# In[46]:


some_labels = df_labels.iloc[:5]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


model.predict(prepared_data)


# In[49]:


some_labels


# In[50]:


list(some_labels)


# ## evaluating the model

# In[63]:


from sklearn.metrics import mean_squared_error
df_predictions = model.predict(df_num_tr)
mse = mean_squared_error(df_labels, df_predictions)
rmse = np.sqrt(mse)


# In[64]:


rmse


# ## cross validation  - evaluation technique

# In[60]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, df_num_tr, df_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[61]:


rmse_scores


# In[65]:


def print_scores(scores):
    print("Scores:" , scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[66]:


print_scores(rmse_scores)


# ##  by random forest

# In[73]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(df_num_tr, df_labels)


# In[74]:


some_data = df.iloc[:5]


# In[75]:


some_labels = df_labels.iloc[:5]


# In[76]:


prepared_data = my_pipeline.transform(some_data)


# In[77]:


model.predict(prepared_data)


# In[78]:


some_labels


# In[79]:


def print_scores(scores):
    print("Scores:" , scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[81]:


rmse_scores


# In[80]:


print_scores(rmse_scores)


# In[82]:


from joblib import dumb, load
dump(model, 'pricing.joblib')


# ## TESTING
# 

# In[87]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[86]:


final_rmse


# In[ ]:




