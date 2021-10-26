#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')
gender_submission = pd.read_csv('titanic/gender_submission.csv')

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
df = pd.concat([train,test])
df = df[['dataset','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]


#%%
df['Age'] = df['Age'].fillna(round(df.groupby('Sex')['Age'].transform('mean')))
df['Embarked'] = df['Embarked'].fillna('S')
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())