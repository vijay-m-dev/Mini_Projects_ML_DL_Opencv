import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#importing dataset 
dataset = pd.read_csv('credit-approval_csv.csv')

print(dataset.head())
print(dataset.describe())
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing

#removing nan rows in categoral columns
dataset[['A2','A3','A8','A11','A14','A15']].describe()
dataset=dataset.dropna(subset=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])

#independent and dependent columns
X=dataset.drop('class',axis=1)
Y=dataset['class']

#Take cae of missing values in numeric columns
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[['A2','A3','A8','A11','A14','A15']])
X[['A2','A3','A8','A11','A14','A15']]=imputer.transform(X[['A2','A3','A8','A11','A14','A15']])
print(X.isnull().sum())

#Numeric variables analysis
Y=pd.get_dummies(Y)['+'].rename("Approve")
X_cont=pd.concat([X[['A2','A3','A8','A11','A14','A15']],Y],axis=1)
print(X_cont.info())

#data distribution
def dist_Plot(col):
    sns.displot(col)
    plt.show()

dist_Plot(X_cont['A2'])
dist_Plot(X_cont['A3'])
dist_Plot(X_cont['A8'])
dist_Plot(X_cont['A11'])
dist_Plot(X_cont['A14'])
dist_Plot(X_cont['A15'])

#Correlation
corr_mat=X_cont[['A2','A3','A8','A11','A14','A15']].corr()
sns.heatmap(corr_mat,vmax=1.0,square=True)

#scatter plot
#sns.set()
sns.pairplot(X_cont,height=3,hue='Approve')

#categorical variable analysis
X_cat=pd.concat([X[['A1','A4','A5','A6','A7','A9','A10','A12','A13']],Y.astype('str')],axis=1)
print(X_cat.info())

fig, ax1=plt.subplots(1,2)
fig, ax2=plt.subplots(1,2)
fig, ax3=plt.subplots(1,2)
fig, ax4=plt.subplots(1,2)
sns.countplot(x="A1",hue="Approve",data=X_cat,ax=ax1[0])
sns.countplot(x="A4",hue="Approve",data=X_cat,ax=ax1[1])
sns.countplot(x="A5",hue="Approve",data=X_cat,ax=ax2[0])
sns.countplot(x="A6",hue="Approve",data=X_cat,ax=ax2[1])
sns.countplot(x="A7",hue="Approve",data=X_cat,ax=ax3[0])
sns.countplot(x="A9",hue="Approve",data=X_cat,ax=ax3[1])
sns.countplot(x="A10",hue="Approve",data=X_cat,ax=ax4[0])
sns.countplot(x="A12",hue="Approve",data=X_cat,ax=ax4[1])
fig.show()
#Removing non important features
X=X.drop(['A1','A12'],axis=1)


#One hot encoding of nominal data

A4=pd.get_dummies(X["A4"],drop_first=True)
A5=pd.get_dummies(X["A5"],drop_first=True)
A6=pd.get_dummies(X["A6"],drop_first=True)
A7=pd.get_dummies(X["A7"],drop_first=True)
A9=pd.get_dummies(X["A9"],drop_first=True)
A10=pd.get_dummies(X["A10"],drop_first=True)
A13=pd.get_dummies(X["A13"],drop_first=True)
X=X.drop(['A4','A5','A6','A7','A9','A10','A13'],axis=1)
X=pd.concat([X,A4,A5,A6,A7,A9,A10,A13],axis=1)

#Splitting train test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#training
#Logistic Regression
'''from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)'''

#Random forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#predictiing
y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
cm=confusion_matrix(Y_test,y_pred)
accuracy_sc=accuracy_score(Y_test,y_pred)
precision_sc=precision_score(Y_test,y_pred)
recall_sc=recall_score(Y_test,y_pred)

#feature importance
importance=classifier.feature_importances_
std=np.std([tree.feature_importances_ for tree in classifier.estimators_],axis=0)
indices=np.argsort(importance)[::-1]
max_cols=10
indices=indices[:max_cols]
print("Top 10 important features")
for i in range(max_cols):
    print("%d. feature %d => %f" % (i+1,indices[i],importance[indices[i]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(max_cols), importance[indices],color="r",yerr=std[indices],align="center")
plt.xticks(range(max_cols),indices)
plt.xlim([-1,max_cols])
plt.show()

