
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
df = pd.read_csv('ALZdataset.csv')
df.head()
df.shape
#replacing the targeg class with numerical values.
df['Group'].replace(['Nondemented','Demented','Converted',],[0,1,2],inplace=True)

#replacing the targeg M/F with numerical values.
df['M/F'].replace(['M','F'],[0,1],inplace=True)
df = df.drop(['Visit','MRI ID','Hand'],axis=1)
df.isnull().sum()
df[['SES','MMSE']].describe()
df['SES'].replace(np.nan,df['SES'].mean(),inplace=True)
df['SES'].isnull().sum()
df['MMSE'].replace(np.nan,df['MMSE'].mean(),inplace=True)
df['MMSE'].isnull().sum()
df.isnull().sum()
df.shape
df.info()
sns.countplot(df['Group'])
sns.pairplot(data=df, vars=['MR Delay','M/F','Age',
                           'EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF'], hue='Group')
plt.show()
pd.crosstab(df['Age'],df['Group']).plot(kind="bar",figsize=(25,8),color=['gold','brown' ])
plt.title('Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
pd.crosstab(df['M/F'],df['Group']).plot(kind="bar",figsize=(10,5),color=['cyan','coral','brown' ])
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease",'Converted'])
plt.ylabel('Frequency')
plt.show()
df['Group'].unique()
pd.crosstab(df['EDUC'],df['Group']).plot(kind="bar",figsize=(10,5),color=['cyan','coral','brown' ])
plt.xlabel('Years Of Education')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease",'Converted'])
plt.ylabel('Frequency')
plt.show()
sns.distplot(df['Age'], color='r')
plt.show()
sns.distplot(df['MMSE'], color='r')
plt.show()
sns.distplot(df['eTIV'], color='r')
plt.show()
df['nWBV'].plot(kind='hist', color='r')
plt.show()
df['ASF'].plot(kind='hist', color='r')
plt.show()
plt.figure(figsize=(12,8))

y = df['Group']
X = df.drop(['Subject ID','Group'],axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#USING RANDOM FOREST CLASSIFIER
vot = RandomForestClassifier(n_jobs=-1,max_features='sqrt')

KNN = KNeighborsClassifier()
xgb = xgb.XGBClassifier()


#TRYING TO GET THE OPTIMAL FEATURES TO BE USED BY THE CLASSIFIER 
param_grid = {
     "n_estimators": [10,100,500],
     "max_depth": [1,5,10,15],
     "min_samples_leaf": [1,2,3,4,5,10,15,20,30,40,50]
 }

GS = GridSearchCV(estimator=vot, param_grid=param_grid,n_jobs=-1, cv=10)
GS.fit(X,y)
print(GS.best_params_)




from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,
                                                    test_size=0.2,random_state=7,shuffle=True)

print(f'X train {len(X_train)} X test {len(X_test)} y train {len(y_train)} y test {len(y_test)}')
vot_model = RandomForestClassifier(n_jobs=-1,max_features='sqrt',
                                  max_depth=10,min_samples_leaf=1,n_estimators=10)
print("VotingClassifier")
vot_model.fit(X_train,y_train)
pred = vot_model.predict(X_test)
vot_model.score(X_test,y_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

print("KNN")

knn_model = KNeighborsClassifier()

knn_model.fit(X_train,y_train)
pred = knn_model.predict(X_test)
knn_model.score(X_test,y_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


print("XGBClassifier")


xgb_model = XGBClassifier()

xgb_model.fit(X_train,y_train)
pred = xgb_model.predict(X_test)
xgb_model.score(X_test,y_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))












