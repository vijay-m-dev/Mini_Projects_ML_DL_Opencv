#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Reading dataset and adding column names
columns=['buying','maint','doors','persons','lus-boot','safety','acc']
dataset = pd.read_csv('car.data',header=None,names=columns)


print(dataset.head())
print(dataset.describe())
print(dataset.info())
#Checking for null values
print(dataset.isnull().sum())
print(dataset.value_counts())


price_values={"buying":{"vhigh":4,"high":3,"med":2,"low":1},
              "maint":{"vhigh":4,"high":3,"med":2,"low":1},
              "doors":{'5more':6},
              "persons":{"more":6},
              "lus-boot":{'big':3,'med':2,'small':1},
              "safety":{'high':3,'med':2,'low':1},
              "acc":{'acc':4,'vgood':3,'good':2,'unacc':1}}

dataset.replace(price_values,inplace=True)
dataset['doors']=dataset['doors'].astype(int)
dataset['persons']=dataset['persons'].astype(int)
print(dataset.info())

Y=dataset['acc']
X_cont=X_cat=dataset

#data distribution
def dist_Plot(col):
    sns.displot(col)
    plt.show()

dist_Plot(X_cont['buying'])
dist_Plot(X_cont['maint'])
dist_Plot(X_cont['doors'])
dist_Plot(X_cont['persons'])
dist_Plot(X_cont['lus-boot'])
dist_Plot(X_cont['safety'])


fig,ax1=plt.subplots(1,2)
fig,ax2=plt.subplots(1,2)
fig,ax3=plt.subplots(1,2)
sns.countplot(x="buying",hue="acc",data=X_cat,ax=ax1[0])
sns.countplot(x="maint",hue="acc",data=X_cat,ax=ax1[1])
sns.countplot(x="doors",hue="acc",data=X_cat,ax=ax2[0])
sns.countplot(x="persons",hue="acc",data=X_cat,ax=ax2[1])
sns.countplot(x="lus-boot",hue="acc",data=X_cat,ax=ax3[0])
sns.countplot(x="safety",hue="acc",data=X_cat,ax=ax3[1])

#Independent and dependent columns
cols=["buying","maint","doors","persons","lus-boot","safety"]
X_set=pd.DataFrame(dataset,columns=cols)
Y_set=pd.DataFrame(dataset,columns=["acc"])

#Splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_set,Y_set,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Random forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#Predicting the test set
y_pred=classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

#iImportant features
importance=classifier.feature_importances_
std=np.std([tree.feature_importances_ for tree in classifier.estimators_],axis=0)
indices=np.argsort(importance)[::-1]
max_cols=6
indices=indices[:max_cols]
print("important features")
for i in range(max_cols):
    print("%d. feature %d => %f" % (i+1,indices[i],importance[indices[i]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(max_cols),importance[indices],color="r",yerr=std[indices],align="center")
plt.xticks(range(max_cols),indices)
plt.xlim([-1,max_cols])
plt.show()
