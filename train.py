#!/usr/bin/env python
# coding: utf-8

# # Machine learning model to accurately classify whether or not the patients in the dataset have diabetes or not

# In[58]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
import  missingno as ms

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

from scipy import stats 

# Warnings
import warnings
warnings.filterwarnings("ignore")


# # Data Visualization

# In[59]:


data = pd.read_csv('Pima_Indian_diabetes.csv')
df = data.copy()
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)
df.head()


# In[60]:


print(df.shape)
df.describe()


# #### We can clearly see that Pregnancies, BloodPressure, SkinThickness and BMI have negative values; Glucose and Insulin have zero values, which are biologically implausible

# In[61]:


df.hist(figsize=(9, 9))
plt.show()


# #### This histogram analysis shows us that BMI, BloodPRessure and Glucose follow Gaussian Distribution which can help us in imputing values in the dataset which we will look at subsequently.
# #### We can also see that Insulin and SkinThickness have a lot of zeros which need to be taken care of

# In[62]:


ms.matrix(df)
plt.show()


# #### This is a graph representing the null values in our dataset, where white lines indicate the null values at that particular index

# In[63]:


df_test_1 = df.copy()
df_test_1.loc[df_test_1['Insulin'] <= 0, 'Insulin'] = np.nan
df_test_1.loc[df_test_1['SkinThickness'] <= 0, 'SkinThickness'] = np.nan
ms.matrix(df_test_1)
plt.show()


# #### After making the zero values of SkinThickness and Insulin as null, we can see that Insulin and SkinThickness has a high percentage of null values

# In[64]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# #### An interesting thing to note here is that SkinThickness and BMI, and Glucose and Insulin have a moderate correlation with each other respectively
# #### Also the correlation between SkinThickness and Outcome is low

# In[65]:


# fig, axs = plt.subplots(2, figsize=(10,10))
# axs[0].scatter(df['SkinThickness'], df['BMI'])
# # axs[0].set_title('SkinThickness')
# axs[0].set_ylabel('SkinThickness')
# axs[0].set_xlabel('BMI') 
# axs[1].scatter(df['Insulin'], df['Glucose'])
# # axs[1].set_title('Insulin')
# axs[1].set_ylabel('Glucose')
# axs[1].set_xlabel('Insulin') 


# # Outlier Detection

# In[66]:


df = data.copy()
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)
df.loc[df['Pregnancies'] < 0, 'Pregnancies'] = np.nan
df.loc[df['Glucose'] <= 0, 'Glucose'] = np.nan
df.loc[df['Insulin'] <= 0, 'Insulin'] = np.nan
df.loc[df['BMI'] <= 7, 'BMI'] = np.nan # least possible known BMI is 7
df.loc[df['BloodPressure'] <= 0, 'BloodPressure'] = np.nan
df.loc[df['Age'] <= 0, 'Age'] = np.nan
df.loc[df['SkinThickness'] <= 0, 'SkinThickness'] = np.nan
df.isnull().sum()


# #### Here we are handling biologically implausible data by making them null

# In[67]:


zscore = 3
print(df.isnull().sum())
df.loc[np.abs(stats.zscore(df['BMI'])) > zscore, 'BMI'] = np.nan
df.loc[np.abs(stats.zscore(df['Glucose'])) > zscore, 'Glucose'] = np.nan
df.loc[np.abs(stats.zscore(df['Insulin'])) > zscore, 'Insulin'] = np.nan
df.loc[np.abs(stats.zscore(df['DPF'])) > zscore, 'DPF'] = np.nan
df.loc[np.abs(stats.zscore(df['BloodPressure'])) > zscore, 'BloodPressure'] = np.nan
df.loc[np.abs(stats.zscore(df['Age'])) > zscore, 'Age'] = np.nan
df.loc[np.abs(stats.zscore(df['Pregnancies'])) > zscore, 'Pregnancies'] = np.nan
df.loc[np.abs(stats.zscore(df['SkinThickness'])) > zscore, 'SkinThickness'] = np.nan
print(df.isnull().sum())
df.describe()


# #### zscore function computes the relative Z-score of the input data, relative to the sample mean and standard deviation which is similar to mahalanobis distance. We have used an absolute threshold of 3 to detect outliers

# # Missing Data Handling 

# In[68]:


def fill_null_with_mean_std(df,feature):
    mean = df[feature].mean()
    std = df[feature].std()
    is_null = df[feature].isnull().sum()
    rand_feature = np.random.randint(mean - std, mean + std, size = is_null)
    age_feature = df[feature].copy()
    age_feature[np.isnan(age_feature)] = rand_feature
    df[feature] = age_feature
    return df

def fill_null_with_mode(df,feature):
    mode = df[feature].value_counts().idxmax()
    df[feature].fillna(mode, inplace = True)
    return df


# In[69]:


gaussian_features = ['Glucose', 'BMI', 'BloodPressure']
df = fill_null_with_mean_std(df,'Glucose')
df = fill_null_with_mean_std(df,'BMI')
df = fill_null_with_mean_std(df,'BloodPressure')
df = fill_null_with_mode(df,'DPF')
df = fill_null_with_mode(df,'Pregnancies')
df = fill_null_with_mode(df,'Age')
df.describe()


# #### For the features Glucose, BMI, and BloodPressure we fill the missing values by taking random values between mean and standard deviation. This can be done because as established earlier these features follow Gaussian Distribution
# #### DPF, Pregnancies and Age are have skewed distribution, so substituting null values with mode makes for a better imputation

# In[70]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[71]:


# Linear regression for SkinThickness and BMI
train_df = df[df['SkinThickness'].notnull()]
x = train_df[['BMI']].values
y = train_df.SkinThickness.values
lr = LinearRegression()
lr.fit(x,y)
x_test = df[np.isnan(df['SkinThickness'])][['BMI']].values 
y_pred = lr.predict(x_test)
SkinThickness_feature = df['SkinThickness'].copy()
SkinThickness_feature[np.isnan(SkinThickness_feature)] = y_pred
df['SkinThickness'] = SkinThickness_feature


# #### We can see that after imputing missing data for all the other features, the correlation between SkinThickness and BMI if 0.6 which is considered high correlation

# In[72]:


plt.scatter(df['SkinThickness'], df['BMI'])
plt.ylabel('SkinThickness')
plt.xlabel('BMI')


# #### Due to high correlation and linearity, imputation of SkinThickness can be performed using Linear Regression with BMI as the feature 

# In[73]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[74]:


# Linear regression for Insulin and Glucose
train_df = df[df['Insulin'].notnull()]
x = train_df[['Glucose']].values
y = train_df.Insulin.values
lr = LinearRegression()
lr.fit(x,y)
x_test = df[np.isnan(df['Insulin'])][['Glucose']].values
y_pred = lr.predict(x_test)
Insulin_feature = df['Insulin'].copy()
Insulin_feature[np.isnan(Insulin_feature)] = y_pred
df['Insulin'] = Insulin_feature


# #### We can see that after imputing missing data for all the other features, the correlation between Insulin and Glucose if 0.56 which is considered high correlation

# In[75]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[76]:


plt.scatter(df['Insulin'], df['Glucose'])
plt.ylabel('Insulin')
plt.xlabel('Glucose')


# #### Simillarly, imputation of Insulin can be performed using Linear Regression with Glucose as the feature 

# # Feature Extraction

# In[77]:


plt.figure(figsize=(20,10))
sns.distplot(df.groupby('Outcome').get_group(1).Age)
sns.distplot(df.groupby('Outcome').get_group(0).Age)


# In[78]:


df['Age_class']=df['Age']/30
df['Age_class']=df['Age'].astype(int)


# #### We can see that there is a age penalty to diabetes for people  whoes age is greater than 30. So in order to quantify this penalty we classify age into two groups ( >=30 and < 30)

# # PCA

# In[79]:


Covariance = df.drop('Outcome', axis=1).cov() 
# Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eig(Covariance)
total = eigen_vals.sum()
pd.set_option('float_format', '{:f}'.format)
print(eigen_vals*100/total)
x=0
a=[]
for i in eigen_vals*100/total:
    x+=i
    a.append(x)
plt.plot(range(1,10),a)
plt.xlabel('No. of Features')
plt.ylabel('Variance Percentage')


# #### More than 99 percent of the variance lies within 5 features as given by eigen values of the covariance matrix. Thus we can reduce the dimensionality of the dataset from 9 (8 + new_feature) to 5 using PCA

# In[80]:


features = df.columns[0:8]
pca = PCA(n_components=5)
x=df.loc[:, features].values
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
df = pd.concat([principalDf, df[['Outcome']]], axis = 1)


# # Model Building

# In[82]:


avg=0
maxx=-1
minn=100
n=100
for _ in range(n):
    train_df, test_df = train_test_split(df)
    X_train = train_df.drop('Outcome', axis=1)
    Y_train = train_df['Outcome']
    X_test = test_df.drop('Outcome', axis=1)
    Y_test = test_df['Outcome']
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    count=0
    acc_log = round (logreg.score(X_test, Y_test) * 100, 2)
    if(acc_log>maxx): 
        maxx=acc_log
    if(acc_log<minn):
        minn=acc_log
    avg += acc_log
print('Minimum accuracy over 100 runs: ', minn)
print('Average accuracy over 100 runs: ', avg/n)
print('Maximum accuracy over 100 runs: ', maxx)


# #### Here we have used Logistic Regression to predict diabetes because we need binary (discreet) output
# #### We are getting average accuracy of 76%
