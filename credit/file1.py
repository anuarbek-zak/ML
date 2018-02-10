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

# print (sampleAccounts[:20])
# print (sampleAccounts.type[:20])
print ('shape: ',sampleAccounts.shape)
# print ('columns: ',sampleAccounts.columns)
# print ('info: ',sampleAccounts.info())
# print ('describe: ',sampleAccounts.describe())
# print ('value_counts 1: ',sampleAccounts.relationship.value_counts())
# print ('value_counts 2: ',sampleAccounts.type.value_counts(normalize=True))
# print (sampleAccounts.sort_values(by=['type','relationship'],ascending=[False,False])[:20])
# print (sampleAccounts[sampleAccounts['type']==14][['credit_limit','bki_request_date']].mean())
# print (sampleAccounts[(sampleAccounts['type']==14) | (sampleAccounts['relationship']==5)]['bki_request_date'].max())
# print (sampleAccounts.loc[:20,'tcs_customer_id':'type'])
#
# print (sampleAccounts.iloc[:20,:6])
# sampleAccounts['type'] = sampleAccounts['type'].map({99:100,9:10})
# print (sampleAccounts['type'][:30])

print (pd.crosstab(sampleAccounts['type'],sampleAccounts['relationship'],margins=True))

# payment_period = sampleAccounts['final_pmt_date'] - sampleAccounts['open_date']
# sampleAccounts.insert(len(sampleAccounts.columns),'payment_period',payment_period)
# print  (sampleAccounts.payment_period)

# print (sampleAccounts.drop(['type','relationship'],axis=1).shape)
#