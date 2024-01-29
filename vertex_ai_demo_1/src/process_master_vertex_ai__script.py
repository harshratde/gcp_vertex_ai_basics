#!/usr/bin/env python
# coding: utf-8

# # Initiate vertex ai model builder

# In[1]:


import os
import argparse
import pandas as pd
# import mlflow
# import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os 
print(os.listdir())

from pathlib import Path


# In[2]:


os.getcwd()


# In[4]:


DATA_IN_PATH = '../DATA/INPUT'
DATA_PROCESSED_PATH = '../DATA/PROCESSED'
DATA_OUT_PATH = '../DATA/OUTPUT'
MODEL_PATH  = '../MODEL'


# In[5]:


Path(DATA_IN_PATH).mkdir( parents=True, exist_ok = True)
Path(DATA_PROCESSED_PATH).mkdir( parents=True, exist_ok = True)
Path(DATA_OUT_PATH).mkdir( parents=True, exist_ok = True)
Path(MODEL_PATH).mkdir( parents=True, exist_ok = True)


# ## 1. Load the data

# In[6]:
credit_df = pd.read_csv("https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
                        header=1,
                        index_col=0)

credit_df.to_csv(f'{DATA_IN_PATH}/credit_inp_data.csv', index=False)
credit_df.to_pickle(f'{DATA_IN_PATH}/credit_inp_data.pkl')

train_df, test_df = train_test_split(credit_df,
                                     test_size=0.25,)

# ## 2. Prepare for training
# Extracting the label column
y_train = train_df.pop("default payment next month")

# convert the dataframe values to array
X_train = train_df.values

# Extracting the label column
y_test = test_df.pop("default payment next month")

# convert the dataframe values to array
X_test = test_df.values

print(f"Training with data of shape {X_train.shape}")

# mlflow.start_run()
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred_df = pd.DataFrame({'pred_val' : y_pred})

print(classification_report(y_test, y_pred))
# Stop logging for this model
# mlflow.end_run()


# In[16]:


ENV_NAME = 'DEV'
PROJECT_NAME = 'DEMO'
PREFIX = 'HR'
ITERATION = 20240128
PREFIC_ITERATION = f'{PREFIX}__{str(ITERATION)}'


GCP_FILE_PATH = f'{ENV_NAME}/{PROJECT_NAME}/{PREFIC_ITERATION}'

print('======================================================================')
print(f'GCP BUCKET PATH : {GCP_FILE_PATH}')
print('======================================================================')


# ## Export data to cloud storage location

# In[19]:


# Will raise an error as the given path is for Google cloud storage, not the local path

# y_pred_df.to_csv(f'{GCP_FILE_PATH}/y_pred.csv' , index=False)
# y_pred_df.to_pickle(f'{GCP_FILE_PATH}/y_pred.pkl')


# In[24]:


# !gsutil -m cp -r {DATA_IN_PATH}/*.csv gs://generic-harsh-buck
get_ipython().system('gsutil -m cp -r {DATA_IN_PATH} gs://generic-harsh-buck')


# In[22]:


DATA_IN_PATH


# In[ ]:




