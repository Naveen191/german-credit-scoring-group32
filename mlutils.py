import pandas as pd
import numpy as np
import copy
from collections import Counter
# from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
# import seaborn as sns
from sklearn.metrics import fbeta_score, f1_score,precision_score,recall_score,accuracy_score, roc_curve, auc,confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
# from credit_actual_data_values import substitute

# define a GradientBoostingclassifier
clf = GradientBoostingClassifier()
# define a SVM classifier
classes = {1: "GoodRisk", 2: "BadRisk"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    # df=pd.read_csv('germanCreditData_hackathon.data', sep=",", header=None)    # do the test-train split and train the model
    df=pd.read_csv('data.csv', sep=",", header=None)    # do the test-train split and train the model
    df.columns = [np.arange(0,df.shape[1])]
    # df.columns
    # df.drop([df.columns[i] for i in [0,5,8,9,10,11,12,13,15,18]],axis = 1, inplace = True)
    last_ix = len(df.columns) - 1
    # X, y  = df.drop([df.columns[i] for i in [0,5,8,9,10,11,12,13,15,18,last_ix]],axis = 1, inplace = True)
    X, y = df.drop(last_ix, axis=1), df[last_ix]
    # d = 0
    # X=X.drop(X.)

    # Categorical features has to be converted into integer values for the model to process. 
    #This is done through one hot encoding.
    # select categorical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    # one hot encode categorical features only
    ct = ColumnTransformer([('o',OneHotEncoder(),cat_ix)], remainder='passthrough')
    X = ct.fit_transform(X)
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    print(X.shape, y.shape, Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    # calculate and print the accuracy score for GaussianNB
    clf_acc = accuracy_score(y_test, clf.predict(X_test))
    y_pred = clf.predict(X_test)
    print(f"Model GradientBoost trained with accuracy: {round(clf_acc, 3)}")
    clf_precision = precision_score(y_test,y_pred, average='micro')
    print(f"Model GradientBoost trained with precision: {round(clf_precision, 3)}")
    clf_recall = recall_score(y_test,y_pred, average='micro')
    print(f"Model GradientBoost trained with recall: {round(clf_recall, 3)}")
    clf_f1 = f1_score(y_test,y_pred, average='macro')
    print(f"Model GradientBoost trained with f1score: {round(clf_f1, 3)}")
   
# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(prediction)
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)