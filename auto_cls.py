# Import the PreprocessingConfig class from the configuration file
from con_ML_preproc import PreprocessingConfig
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


# Read the data
data = pd.read_csv(r'train.csv')

X = data.drop('Survived', axis=1)
y = data['Survived']

# Create an instance of the PreprocessingConfig class
config = PreprocessingConfig(X)


# Apply the preprocessing steps
data_processed = config.convert_to_object()
data_processed = config.handle_missing_values()
data_processed = config.cap_outliers()
data_processed = config.categorical_encoding()
data_processed = config.transform_data()



# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_processed, y, test_size=0.30, random_state=90)


rf = RandomForestClassifier()
# Fit the pipeline to the training data
rf.fit(X_train, y_train)

# Predict using the pipeline
y_pred = rf.predict(X_test)

# Evaluate the pipeline using metrics such as accuracy, precision, recall, f1, and roc_auc
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("F1 Score: ", f1)
print("ROC AUC: ", roc_auc)
