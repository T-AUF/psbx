# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:41:01 2021

@author: 9111650T
"""

import pretraitement as pt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import warnings
warnings.filterwarnings("ignore")

modeles_a_tester = ["RL", "NB", "RF", "Arbre", "xgb"]

dico_modeles = {"RL" : LogisticRegression(),
                "NB" : GaussianNB(),
                "RF" : RandomForestClassifier(), 
                "Arbre" : DecisionTreeClassifier(), 
                "xgb" : xgb.XGBClassifier()
                }

dico_params = { "RL" : {'C':[0.3,0.6,0.8,1,1.2,1.5,2]},
                "NB" : {},
                "RF" : {'n_estimators' : [10,25,50], 'max_depth' : [6,8,10,12], 'min_samples_leaf': [25,50] },
                "Arbre" : {'max_depth' : [6,8,10,12], 'min_samples_leaf': [25,50] },
                "xgb" : {}     
               }
   

#df = pt.Grande_Prepa()
#df = pd.read_csv("Merged_MSC_less_columns.csv", sep=';')
df = pt.Prepa_Dataiku()
df['SG_DIST'].fillna(df['SG_DIST'].median(), inplace=True)
print(df.columns[df.isnull().any()])

#label column : what to predict
target = "label"
df['label'] = df['label'].replace({2:0})
print(len(df), df['label'].sum())
#0 = any doubt, 1 = doubt cars only
print(df['label'].unique())


#features for prediction
predictors = df.columns.tolist()

#Les 3 lignes ci-dessous doivent disparaitre. Tous les traitements spéciaux doivent être faits à un moment.
for p in pt.special:
    print(p)
    predictors.remove(p)

X = df[predictors]#.values
y = df[target]#.values

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

df = df.reset_index()

def Scores(y_test, y_pred, modele='modele'):
    print("###########", modele, "############")
    print("Confusion_matrix:", metrics.confusion_matrix(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision_hesite:", metrics.precision_score(y_test, y_pred, pos_label=1))
    print("Recall_hesite:", metrics.recall_score(y_test, y_pred, pos_label=1))
    print("Precision_hesite_pas:", metrics.precision_score(y_test, y_pred, pos_label=0))
    print("Recall_hesite_pas:", metrics.recall_score(y_test, y_pred, pos_label=0))
    print("ROC_AUC:", metrics.roc_auc_score(y_test, y_pred))
    #print("ROC_Curve:", metrics.roc_curve(y_test, y_pred))
    print()

def Test_Modele(modele, X_train, X_test, y_train, y_test, nom_modele=''):
    #modele.fit(X_train, y_train)
    
    mon_recall = make_scorer( metrics.recall_score, pos_label=1)
    #    retrun metrics.recall_score(y_test, y_pred, pos_label=1)
    
    
    clf = GridSearchCV(modele, dico_params[nom_modele], refit = True, scoring=metrics.f1_score)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    #print(clf.cv_results_)
    y_pred = clf.best_estimator_.predict(X_test)
    Scores(y_test, y_pred, modele = nom_modele)
    #return clf.best_estimator_
 
    

for modele in modeles_a_tester:
#for modele in ['RL']:
    Test_Modele(dico_modeles[modele], X_train, X_test, y_train, y_test, modele)


# =============================================================================
# 
# #Regression Logistique
# RL = LogisticRegression()
# RL.fit(X_train, y_train)
# y_pred = RL.predict(X_test)
# Scores(y_test, y_pred, modele = 'Régression logistique')
# print('coefficients de la regression logisitique')
# coeffs = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(RL.coef_))], axis = 1)
# coeffs.columns = ['nom','coef']
# coeffs.sort_values('coef', inplace=True)
# print(coeffs.head())
# print(coeffs.tail())
# print(coeffs.columns.values)
# 
# #Arbre de decision
# dt = DecisionTreeClassifier(max_depth=20, criterion="gini")
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# Scores(y_test, y_pred, modele="Arbre de decision")
# 
# 
# #SVM
# #dt = SVC()
# #dt.fit(X_train, y_train)
# #y_pred = dt.predict(X_test)
# #Scores(y_test, y_pred, modele="SVC")
# 
# 
# #Naive Bayes
# dt = GaussianNB()
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# Scores(y_test, y_pred, modele="Naive Bayes")
# 
# 
# #XGBoost
# dt = xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.3, \
#                 max_depth = 20, alpha = 50, n_estimators = 50)
# dt.fit(X_train, y_train)
# 
# y_pred = dt.predict(X_test)
# Scores(y_test, y_pred, modele="XGBoost")
# y_pred = dt.predict(X_train)
# Scores(y_train, y_pred, modele="XGBoost")
# 
# 
# 
# 
# #Random Forest
# dt = RandomForestClassifier(max_depth=20, n_estimators = 50, min_samples_leaf=50, criterion="gini")
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# Scores(y_test, y_pred, modele="Random Forest")
# 
# #Pour tous les modeles, il faut chercher le meilleur jeu d'hyperparametres en faisant un GridSearchCV et cross validation
# 
# #Il faut continuer à eliminer des variables inutiles
# #    soit par l'intelligence
# #    soit par https://machinelearningmastery.com/an-introduction-to-feature-selection/
# 
# 
# =============================================================================
