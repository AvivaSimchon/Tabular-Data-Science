import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class CreateModels:

    def __init__(self,dataset):
        self.dataset = dataset

    def create_model(self):
        #print('create model')
        features_names = self.dataset.columns[:-1]

        # The target variable is the last column in the df
        Y = self.dataset.iloc[:,-1]
        X = self.dataset[features_names]
        # Split the data into train and test data:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state = 42)
        # Build the model with the random forest regression algorithm:
        # model = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
        # XGBOOST
        model = XGBClassifier()
        model.fit(X_train, Y_train)


        # predicting the test set results
        y_pred = model.predict(X_test)
        scores = model.predict_proba(X_test)

        # Calculating the accuracies
        #print("Training accuracy :", model.score(X_train, Y_train))
        #print("Testing accuracy :", model.score(X_test, Y_test))

        # classification report
        #print(classification_report(Y_test, y_pred))

        # confusion matrix
        #print(confusion_matrix(Y_test, y_pred))

        #save model as pickle file
        # with open('model_wine.pkl', 'wb') as f:
        #     pickle.dump(model, f)

        return model
