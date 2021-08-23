import pandas as pd

import numpy as np

import pickle

loan_data = pd.read_csv("data.csv")

#create a copy
loan_prep = loan_data.copy()

#checking missing values

loan_prep.isnull().sum(axis=0)

#we can solve it by using mean for numeric and for categorical value we can use mode

#we will here directly drop null values bcoz it will drop very less records

loan_prep = loan_prep.dropna()

#create dummy variables for categorical values before that lets drop useless columns

loan_prep = loan_prep.drop(['gender'], axis = 1)

loan_prep.dtypes
#as we can see for categorical columns married and status columns are of type object and ch is of type float but we can directly use get dummies funcn bcoz the ch has only 1 and 0 values no need to convert ch in object or categorical type

loan_prep = pd.get_dummies(loan_prep,drop_first = True)

#Normalize the data(Income and Loan Amount) Using StandardScaler

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

loan_prep['income'] = scalar.fit_transform(loan_prep[['income']])

loan_prep['loanamt'] = scalar.fit_transform(loan_prep[['loanamt']])

#Create X and Y

Y = loan_prep['status_Y']

X = loan_prep.drop(['status_Y'],axis=1)

#split the data in train and test

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234,stratify = Y)
#stratify here means it can be a probability that our train only contains all yes or all no ot vice versa for split as well so in that case we use stratify it will divide yes and no in equal proportions like if we use 50% train test split it will multiply 0.5*all yes and all no to get 50% of yes and no

#IMPORT random forest TREE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rfc =RandomForestClassifier(random_state=1234)
rfc.fit(X_train, Y_train)


# Test the model
Y_predict3 = rfc.predict(X_test)

# Evaluate the model

cm3 = confusion_matrix(Y_test, Y_predict3)
rfc_score = rfc.score(X_test, Y_test)

cross_score_rfc = cross_val_score(rfc,X,Y,cv=10)
print("Accuracy of RFC IS:",rfc_score*100)
print("Cross validation rfc is",np.mean(cross_score_rfc)*100)


#HYPERPARAMETER TUNING

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV


# Define parameters for Random Forest
rfc_param = {'n_estimators':[10,15,20], 
            'min_samples_split':[8,16],
            'min_samples_leaf':[1,2,3,4,5]
            }

# The parameters results in 3 x 2 x 5 = 30 different combinations
# CV=10 for 30 different combinations mean 300 jobs/model runs

rfc_grid = GridSearchCV(estimator=rfc, 
                        param_grid=rfc_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search
rfc_grid_fit = rfc_grid.fit(X, Y)

# Get the results of the GridSearchCV
cv_results_rfc = pd.DataFrame.from_dict(rfc_grid_fit.cv_results_)

rfc_top_rank = cv_results_rfc[cv_results_rfc['rank_test_score'] == 1]


# Print the best parameters of the Random Forest Classifier
print('\n The best Parameters are : ')
print(rfc_grid_fit.best_params_)

rfc_mean = cv_results_rfc[['mean_test_score','mean_train_score']].mean()

print ('rfc_mean',rfc_mean)

print (f'Accuracy - : {rfc_grid_fit.score(X_test,Y_test):.3f}')

##print("Accuracy is",rfc_grid_fit.score(X_test,Y_test)*100)