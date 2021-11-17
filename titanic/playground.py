#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

#%%
# EDA
train.describe()

# 1. See M/F survival rates: F,M
# 2. See survival rates per class: 1,2,3
# 3. See survival rates per Embarked: C,Q,S

train.groupby('Sex')['Survived'].describe()
train.groupby('Pclass')['Survived'].describe()
train.groupby('Embarked')['Survived'].describe()

#%%
# correlation map
corr = train.corr()
sns.heatmap(corr,vmin=-1, vmax=1, cmap=sns.diverging_palette(22,222))

#%%
train['dataset'] = 'train'
test['dataset'] = 'test'
df = pd.concat([train,test])
df = df[['dataset','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

df['Age'] = df['Age'].fillna(round(df.groupby('Sex')['Age'].transform('mean')))
df['Embarked'] = df['Embarked'].fillna('S')
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df = pd.get_dummies(df).drop(columns = 'dataset_train')

#%%
# The feature matrix X should be standardized before fitting. 
# This ensures that the penalty treats features equally.

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix

def compute_accuracy(df, real, pred):
    return sum(df[real]==df[pred])/len(df)


X_train = df[df['dataset_test']==0].drop(columns = ['dataset_test','Survived'])
y_train = df[df['dataset_test']==0]['Survived']

# fit model
model = LogisticRegression(max_iter=400)
model = model.fit(X_train, y_train)
y_fitted = model.predict(X_train)
y_prob = model.predict_proba(X_train)

# obtain result
df_fitted = X_train.copy()
df_fitted['Actual'] = y_train
df_fitted['Predicted'] = y_fitted

# model parameters
model.coef_

# evaluation
compute_accuracy(df_fitted,'Actual','Predicted')
confusion_matrix(y_train,y_fitted)
print(classification_report(y_train, y_fitted))
#%%
# sklearn does not have p values...
# try statsmodels, gives summary report
from sklearn import pipeline
import statsmodels