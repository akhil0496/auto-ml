# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:45:20 2023

@author: akhil
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class PreprocessingConfig:
    def __init__(self, data):
        self.data = data.copy()

    def convert_to_object(self):
        for col in self.data.columns:
            if (self.data[col].dtype in ['int64', 'float64']) and (self.data[col].nunique() < 25):
                self.data[col] = self.data[col].astype('object')
        return self.data

    def handle_missing_values(self):
        for col in self.data.columns:
            if (self.data[col].isnull().sum()/self.data.shape[0])*100 > 50:
                self.data = self.data.drop([col], axis = 1)
            else:
                if self.data[col].dtype in ['float64','int64']:
                    self.data[col].fillna(self.data[col].median(), inplace = True)
                elif self.data[col].dtype == 'object':
                    self.data[col].fillna(self.data[col].mode()[0], inplace = True)
        return self.data

    def cap_outliers(self):
        for col in self.data.select_dtypes(include=[np.number]).columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5*IQR
            lower = Q1 - 1.5*IQR
            self.data.loc[(self.data[col] > upper),col] = upper
            self.data.loc[(self.data[col] < lower),col] = lower
        return self.data

    def categorical_encoding(self):
        for col in self.data.select_dtypes(include=['object','category']).columns:
            if len(self.data[col].value_counts()) < 10:
                self.data = pd.get_dummies(self.data, columns=[col],prefix=[col])
            else:
                self.data[col] = self.data[col].astype('category')
                self.data[col] = self.data[col].cat.codes
        return self.data

    def transform_data(self):
        pt = PowerTransformer(method='yeo-johnson')
        self.data[self.data.select_dtypes(include=[np.number]).columns] = pt.fit_transform(self.data[self.data.select_dtypes(include=[np.number]).columns])
        return self.data

#data = pd.read_csv('train.csv')
#data.info()

# Usage
#config = PreprocessingConfig(data)
#config.convert_to_object()
#config.handle_missing_values()
#config.cap_outliers()
#config.categorical_encoding()
#config.transform_data()









