import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV

import pickle

loan_data = pd.read_csv("G:\\CourseDown.Com.completedatascienceandmachinelearningusingpython\\15 Logistic Regression\\157 01Exercise1.csv")

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


#create the support vector classifier

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train,Y_train)

Y_predict2 = svm.predict(X_test)

#Create a confusion matrix

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(Y_test,Y_predict2)

svc_score = svm.score(X_test,Y_test)


# define parameters for Support Vector Classifier
svc_param = {'C':[0.01, 0.1, 0.5, 1, 2, 5, 10,], 
            'kernel':['rbf', 'linear','sigmoid'],
            'gamma':[0.1, 0.25, 0.5, 1, 5,]
            }

# The parameters results in 7 x 2 x 5 = 70 different combinations
# CV=10 for 70 different combinations mean 700 jobs/model runs

svc_grid = GridSearchCV(estimator=svm, 
                        param_grid=svc_param,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1,
                        return_train_score=True)

# Fit the data to do Grid Search for Support Vector
svc_grid_fit = svc_grid.fit(X, Y)

# Get the Grid Search results for Support Vector
cv_results_svc = pd.DataFrame.from_dict(svc_grid_fit.cv_results_)

svc_top_rank = cv_results_svc[cv_results_svc['rank_test_score'] == 1]

svc_mean = cv_results_svc[['mean_test_score','mean_train_score']].mean()

print('svc_mean\n',svc_mean)

print("Accuracy is", svc_grid_fit.score(X_test, Y_test)*100)

pickle_out = open("loansvc_pkl" , "wb")
pickle.dump(svm, pickle_out)
loaded_model = pickle.load(open("loansvc_pkl" , "rb"))
result = loaded_model.score(X_test, Y_test)
print(result)