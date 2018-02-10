from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import os

pathCustomers = os.path.abspath("SAMPLE_CUSTOMERS.csv")
pathAccounts = os.path.abspath("SAMPLE_ACCOUNTS.csv")

sampleCustomers  = read_csv(pathCustomers,';')
sampleAccounts  = read_csv(pathAccounts,';')

print (sampleCustomers.head())
print (sampleAccounts.shape)

sampleAccounts.tcs_customer_id = sampleAccounts[['tcs_customer_id','open_date','final_pmt_date','credit_limit','currency']].drop_duplicates()

print (sampleAccounts.isnull().sum())
sampleAccounts.replace('?', -99999, inplace=True)
sampleAccounts.final_pmt_date[sampleAccounts.final_pmt_date.isnull()] = sampleAccounts.fact_close_date[sampleAccounts.final_pmt_date.isnull()].astype(float)
sampleAccounts.fillna(0,inplace=True)
print (sampleAccounts.isnull().sum())


