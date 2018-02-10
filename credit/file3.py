from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

sampleCustomers  = read_csv("SAMPLE_CUSTOMERS.csv",';')
sampleAccounts  = read_csv("SAMPLE_ACCOUNTS.csv",';')

print (sampleCustomers.head())