from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
import time


model_params = {
    'knn': {
        'model': KNeighborsClassifier(),
        'params' : {
#            'n_neighbors': [3, 5, 11, 19],
#            'weights': ['uniform', 'distance'],
#            'metric': ['euclidean', 'manhattan']
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params' : {
        }
    },
    'svm': {
        'model': svm.SVC(), #https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
        'params':{
#            'C': [0.1, 1, 10, 100, 1000],
#            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#            'kernel': ['rbf','linear']
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
#            'max_leaf_nodes': list(range(2, 20)),
#            'min_samples_split': [2, 3, 4]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(), #https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
        'params': {
#            'n_estimators': [1, 5, 10, 20, 40, 100, 200],
#            'max_features': ['auto', 'sqrt', 'log2'],
#            'max_depth' : [4,5,6,7,8],
#            'criterion' :['gini', 'entropy']
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(),
        'params': {
#            'n_estimators': [10, 20, 100, 200],
#            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5]
        }
    },
#    'GradientBoostingClassifier' : {
#        'model': GradientBoostingClassifier(),
#        'params': {
#            "loss":["deviance"],
#            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#            "min_samples_split": np.linspace(0.1, 0.5, 12),
#            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#            "max_depth":[3,5,8],
#            "max_features":["log2","sqrt"],
#            "criterion": ["friedman_mse",  "mae"],
#            "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#            "n_estimators":[10]
#        }
#    },
    'LinearDiscriminantAnalysis' : {
        'model': LinearDiscriminantAnalysis(),
        'params': {
    }
    },
    'MLPClassifier' : {
        'model': MLPClassifier(),
        'params': {
#       'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#       'activation': ['tanh', 'relu'],
#       'solver': ['sgd', 'adam'],
#       'alpha': [0.0001, 0.05],
#       'learning_rate': ['constant','adaptive'],
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(),
        'params': {
#            'penalty': ['l1', 'l2'],
#            'C':[0.001, 0.01, 1, 10, 100, 1000, 10000, 100000]
        }
    }
}

'''
model_params = {
    'logistic_regression' : {
        'model': LogisticRegression(multi_class='auto'),
        'params': {
            'penalty': ['l1', 'l2'],
            'C':[0.001, 10000, 100000]
        }
    }
}
'''
featureset='projX_H32'
scores = []
df = pd.read_csv('Data_csv//Train_' + featureset + '.csv') #,sep=';')
for c in df.columns[df.dtypes == object]: # df.dtypes == 'object'
    df[c] = df[c].astype('category').cat.codes

X_train=df.iloc[:,:-1]
y_train=df.iloc[:,-1]

df = pd.read_csv('Data_csv//Test_' + featureset + '.csv') #,sep=';')
for c in df.columns[df.dtypes == object]: # df.dtypes == 'object'
    df[c] = df[c].astype('category').cat.codes

X_test=df.iloc[:,:-1]
y_test=df.iloc[:,-1]

for model_name, mp in model_params.items():
    start_time_run=time.time()

    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    elapsed_time_run = time.time() - start_time_run

    y_pred=clf.predict(X_test)

    test_acc=metrics.accuracy_score(y_test, y_pred)

    print("clf", clf, "test_acc", test_acc, "time", elapsed_time_run)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'test_acc': test_acc,
        'best_params': clf.best_params_,
        'elapsed_time_run': elapsed_time_run
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params', 'test_acc', 'elapsed_time_run'])
df.to_csv("Results//results_" + featureset +".csv")

