# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:02:51 2020

@author: ajay
"""

#%% Importing library
import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.decomposition import pca
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

#%% Loading raw data from dataset folder
coust_behav=pd.read_table('Problem2_Site_BrowingBehavior')
coust_behav.columns=['Timestamp','UserID','website_section_visited']
coust_behav=coust_behav.sort_values(['UserID'])

final_conv=pd.read_table('Problem2_FInalConversions')
final_conv.columns=['Timestamp','UserID','Products_Purchased','Cart_Value']
final_conv=final_conv.sort_values(['UserID'])

# Total number of userid data in both datasets
print('Total number of UserID in coustomer behaviour dataset:',coust_behav['UserID'].nunique())
print('Total number of UserID in final conversion dataset:', final_conv['UserID'].nunique())

# total website section in behaviour dataset
sections=list(coust_behav['website_section_visited'].unique())
uid_coust_behav=coust_behav['UserID'].unique()

# Random shuffling of userid of coustomer behavior dataset
np.random.seed(32)
np.random.shuffle(uid_coust_behav)

#%% Creating features
# creating new dataframe with 17 new feature which are count of website section visited
data=pd.DataFrame(columns=sections)
a1=data.copy()

# Due to system constrain (taking huge amount of time), running the loop for 5000 datapoints. 
# It means a new dataframe with new feature is generated for 5000 userid randomly selected from 
# behaviour dataset.
for i in range(0,5000):
    sel_uid=coust_behav[coust_behav['UserID']==uid_coust_behav[i]].groupby(
        ['UserID','website_section_visited'],as_index=False)['Timestamp'].count()
    df=pd.DataFrame(data=[list(sel_uid['Timestamp'].values)],columns=sel_uid['website_section_visited'])    
    df['UserID']=uid_coust_behav[i]
    k=pd.concat([a1,df],axis=0)
    data=data.append(k)
data=data.fillna(0) # Removing NaN and replacing with zeros

# Adding one column in new dataframe which tells wheteher user has purchased the product or not.
# This column is having binary data. '1' means user has purchased the product and '0' means
# user has not purchased.
data_uid=data['UserID'].unique()
final_conv_uid=final_conv['UserID']
conversion=[]
for i in range(0,5000):
    if data_uid[i] in list(final_conv_uid):
        conversion.append(1)
    else:
        conversion.append(0)
        
data['conversion']=conversion
data=data.reset_index(drop=True)
_, counts = np.unique(conversion, return_counts=True)
print('Number of users purchased the product:',counts[1])
print('Number of users not purchased the product:',counts[0])

# Saving dataframe to reduce time for running the script
data.to_csv('data.csv')

#%% Exploratory data analysis
# This section will describe the coustomer behavior dataset
# TSNE plot for multidimensional visualization.
def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Not purchased')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.5, label='Purchased')
    plt.title(name,fontsize=16,fontweight='bold')
    plt.legend(loc='best')
    plt.show()

rep_x = data[data.columns.drop(['UserID','conversion'])].values
rep_y = data['conversion'].values
tsne_plot(rep_x, rep_y, 'TSNE plot')

# PCA- Principal component analysis for multi dimensional visualaization
xpca=pca.PCA(n_components=2).fit_transform(rep_x)
# scatter plot of principal components
plt.scatter(xpca[rep_y==0][:,0],xpca[rep_y==0][:,1])
plt.scatter(xpca[rep_y==1][:,1],xpca[rep_y==1][:,0])
plt.title('PCA plot',fontsize=16,fontweight='bold')
plt.legend(['Not purchased','Purchased'])
plt.xlabel('1st Princial Component',fontsize=12)
plt.ylabel('2nd Princial Component',fontsize=12)

#%% Modelling the problem.
# This problem is solved using various classification algorithm like naive-bayes,KNN,adabosst,xcgboost,
# Decesion tree etc. and later choosen best among them for computation of scores. 
y=data['conversion'].values
x=data[data.columns.drop(['UserID','conversion'])]

# Spliting dataset into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=32)
print('Training data shape:',np.shape(x_train))
print('Test data shape:',np.shape(x_test))
scaler=StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
y_train=np.reshape(y_train,(len(y_train),-1))
y_test=np.reshape(y_test,(len(y_test),-1))

# Various classifiers are tried with there default parameter
classifier=[GaussianNB,KNeighborsClassifier,AdaBoostClassifier,
            DecisionTreeClassifier,XGBClassifier]
def prediction(clf,x,y,k):
    pred=clf.predict_proba(x)
    pred_1=clf.predict(x)
    print('Test Accuracy:',metrics.accuracy_score(y,pred_1))
    cv=KFold(n_splits=k,shuffle=True,random_state=33)
    scores = cross_val_score(clf, x, y, cv=cv)
    print("Average coefficient of determination using 5-foldcrossvalidation:",np.mean(scores))        
    yes_auc = roc_auc_score(y, pred[:,1])
    print('ROC AUC=%.3f' % (yes_auc))
    return pred

def best_classifer(x_tr,y_tr,x_te,y_te,k):    
    for i in range(len(classifier)):        
        scaler = StandardScaler().fit(x_tr)
        x_tr=scaler.transform(x_tr)
        print(classifier[i])
        clf=classifier[i]().fit(x_tr,y_tr)
        pred_1=clf.predict(x_tr)
        print('Training Accuracy:',metrics.accuracy_score(y_tr,pred_1))
        _=prediction(clf,x_te,y_te,k)

# Model training using various classifiers
best_classifer(x_train,y_train,x_test,y_test,5)

#%% Tuning model
# Tuneing hyperparameter for best classifier. XGBoost has highest accuracy and AUC value
param_dist = {'n_estimator':[100,1000,2000,3000],'learning_rate':[0.01,0.1,0.2,0.3]}
g_search = GridSearchCV(XGBClassifier(),param_grid = param_dist,
                            n_jobs=-1)
g_search.fit(x_train,y_train)
print(g_search.best_score_)
print(g_search.best_params_)

# Best paramater selected
clf=XGBClassifier(n_estimator=100,learning_rate=0.01).fit(x_train,y_train)
print('Accuracy on train data:',metrics.accuracy_score(y_train,clf.predict(x_train)))
print('Accuracy on test data:',metrics.accuracy_score(y_test,clf.predict(x_test)))
clf.predict()
# Proabiblity score if user buys product or not.
score=pd.DataFrame(clf.predict_proba(x_test),columns=['Not_purchased','Purchased'])
score.to_csv('result_score.csv')

#%% Conclusion
# This model has accuracy of 93% on test dataset for determining whether user will
# buy the product or not. This accuracy might change if tested on whole dataset. 
# Accuracy can be improved by adding other features like total time spent on brousing etc.
# This model can be further used to target some specific users for selling product based on 
# user product searched.
