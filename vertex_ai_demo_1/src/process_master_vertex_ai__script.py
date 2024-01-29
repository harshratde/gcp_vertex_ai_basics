#!/usr/bin/env python
# coding: utf-8

# # Initiate vertex ai model builder

import sys
import os
import argparse
import pandas as pd
# import mlflow
# import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from icecream import ic
from pathlib import Path


def custom_output_function(*args):
    # Join the arguments into a string and enclose in double quotes
    output = ' '.join(str(arg) for arg in args)
    output = f'"{output}"'  # Add double quotes
    print(output)

# Configure ice cream to use the custom output function
ic.configureOutput(outputFunction=custom_output_function)

ic(os.listdir())
ic(os.getcwd())


ENV_EXEC = 'ML-SERVER'

# ---- LOAD FROM config_env.yml ----------------
def read_config_env():
    
    BUCKET_NAME = "generic-harsh-buck"
    
    return BUCKET_NAME


# ---- LOAD FROM config_run.yml ----------------
def read_config_run():
    ENV_NAME = 'DEV'
    PROJECT_NAME = 'DEMO'
    PREFIX = 'HR'
    ITERATION = 20240128
    PREFIC_ITERATION = f'{PREFIX}__{str(ITERATION)}'

    GCP_FILE_PATH = f'{ENV_NAME}/{PROJECT_NAME}/{PREFIC_ITERATION}'
    print('======================================================================')
    print(f'GCP BUCKET PATH : {GCP_FILE_PATH}')
    print('======================================================================')
    
    return GCP_FILE_PATH

    
BUCKET_NAME   = read_config_env()
GCP_FILE_PATH = read_config_run()
    
    
if ENV_EXEC == 'ML-SERVER' :
    
    GCP_PREFIX = 'gs:/'
    
    GCP_ROOT_PROJECT_PATH = f'{GCP_PREFIX}/{BUCKET_NAME}/{GCP_FILE_PATH}'
    
    DATA_IN_PATH        = f'{GCP_ROOT_PROJECT_PATH}/INPUT'
    DATA_PROCESSED_PATH = f'{GCP_ROOT_PROJECT_PATH}/PROCESSED'
    DATA_OUT_PATH       = f'{GCP_ROOT_PROJECT_PATH}/OUTPUT'
    MODEL_PATH          = f'{GCP_ROOT_PROJECT_PATH}/MODEL'
    
else:
    DATA_IN_PATH = '../DATA/INPUT'
    DATA_PROCESSED_PATH = '../DATA/PROCESSED'
    DATA_OUT_PATH = '../DATA/OUTPUT'
    MODEL_PATH  = '../MODEL'
    
    Path(DATA_IN_PATH).mkdir( parents=True, exist_ok = True)
    Path(DATA_PROCESSED_PATH).mkdir( parents=True, exist_ok = True)
    Path(DATA_OUT_PATH).mkdir( parents=True, exist_ok = True)
    Path(MODEL_PATH).mkdir( parents=True, exist_ok = True)

print('======================================================================')
ic(DATA_IN_PATH)
ic(DATA_PROCESSED_PATH)
ic(DATA_OUT_PATH)
ic(MODEL_PATH)
print('======================================================================')




# ## 1. Load the data

credit_df = pd.read_csv("https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
                        header=1,
                        index_col=0)
# ALTERNATE PATH
# f'/gcs/{BUCKET_NAME}/{GCP_FILE_PATH}'

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

ic(classification_report(y_test, y_pred))
# Stop logging for this model
# mlflow.end_run()

y_pred_df.to_csv(f'{DATA_OUT_PATH}/y_pred.csv' , index=False)
y_pred_df.to_pickle(f'{DATA_OUT_PATH}/y_pred.pkl')



# ## Export data to cloud storage location

# Will raise an error as the given path is for Google cloud storage, not the local path
# y_pred_df.to_csv(f'{GCP_FILE_PATH}/y_pred.csv' , index=False)
# y_pred_df.to_pickle(f'{GCP_FILE_PATH}/y_pred.pkl')

"""
# !gsutil -m cp -r {DATA_IN_PATH}/*.csv gs://generic-harsh-buck
# get_ipython().system('gsutil -m cp -r {DATA_IN_PATH} gs://generic-harsh-buck')
"""




