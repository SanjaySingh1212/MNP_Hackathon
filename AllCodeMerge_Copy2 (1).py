import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # For creating plots
from sklearn import metrics
import os
dict = {}
# This path should be user specific based upon file locaiton where Input data is located
os.chdir('C:\\Users\\s114sing\\OneDrive - Nokia\\Training Material\\Hackathon\\telecom-customer')
data = pd.read_csv('Telecom_Manipulated_Data.csv')
print(data.dtypes)
print(data.describe(include = [np.number]))
col = list (data.describe(include = [np.number]))
print (col)
print(data)
print(data.describe())
print(data.corr())
y = data['churn'].values
X = data.drop(columns= ['churn'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#================================================================================================================
# [Running logistic regression model]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)
from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print("Logistic Regression Prediction Score")
LogisticReg = metrics.accuracy_score(y_test, prediction_test)
print (metrics.accuracy_score(y_test, prediction_test))
dict['Logistic Regression']= LogisticReg
# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
#Printing the Weightage of Column for Prediction. User can uncomment below two lines of Code for getting the Result
#print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))
#print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))
#
#Code for Splitting into four Quadrants
#0-0 True Positive - Actual and Predicted values are Correct
#0-1 False Negative - Actual Values are Correct and Predicted values are wrong
#1-0 False Positive - Actual Values are Wrong and Predicted values are correct 
#1-1 True Negative - Actual Values and Predictive values are Incorrect
print ("Confusion Matrix for Logistic Regression")
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,prediction_test)
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True)
plt.show()
#If user wanted to print the Classification Report, Uncomment the below 2 lines of Code and Execute it
#from sklearn.metrics import classification_report
#print(classification_report(y_test,prediction_test))
#===================================================================================================================
# [Running Decision Tree Model]
# Splitting the data set into training and testing datasets
# We use the module model_selection
# Within this module, we use the method called test train split
import sklearn.model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X,y,test_size=0.2, random_state = 42)
#Building Decision Tree Model 
import sklearn.tree as tree
clf = tree.DecisionTreeClassifier(max_depth = 6, random_state=42)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
# Finding the predicted probabilities
clf.predict_proba(X_test)
# Computing ROC AUC Score from  the actual labels and the predicted probabilities from the decision tree model
metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
# Performing grid search cross validation for fine tuning 
clf = tree.DecisionTreeClassifier(max_depth = 4, random_state=42)
mod = ms.GridSearchCV(clf, param_grid={'max_depth':[2,3,4,5,6]})
mod.fit(X_train,y_train)
#Printing the Best Estimator
mod.best_estimator_
#Printing the Best Score
mod.best_score_
#Training
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
result = model.fit(X_train, y_train)
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print("Decision Tree Classifier Prediction Score")
DecisionTree = metrics.accuracy_score(y_test, prediction_test)
print(metrics.accuracy_score(y_test, prediction_test))
dict['Decision Tree'] = DecisionTree
#
#Code for Splitting into four Quadrants
#0-0 True Positive - Actual and Predicted values are Correct
#0-1 False Negative - Actual Values are Correct and Predicted values are wrong
#1-0 False Positive - Actual Values are Wrong and Predicted values are correct 
#1-1 True Negative - Actual Values and Predictive values are Incorrect
print ("Confusion Matrix for Decision Tree")
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,prediction_test)
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True)
plt.show()
#If user wanted to print the Classification Report, Uncomment the below 2 lines of Code and Execute it
#from sklearn.metrics import classification_report
#print(classification_report(y_test,prediction_test))
#
# This code is meant for printing the Attributes as Decision Tree with highest weightage / Prediction descending from Top to Bottom
#Code for Printing the Decision Tree
#import pydotplus
# This path should be user specific based upon the file
#os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin"
# Use the trained model which is stored in the object clf to create a representation feature_names is the column names in my predictor matrix 
# The classes are the two classes of the target variable 
#dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=["0","1"], filled=True, rounded= True, special_characters=True, proportion= True)
# Create a graph representation from dot_data
#graph = pydotplus.graph_from_dot_data(dot_data)
# Use the image module to visualize the tree that was just built. 
#from IPython.display import Image
# Printing the Decision Tree
print ("Attributes affecting the Customer to Port Out from Existing Service Provider")
#Image(graph.create_png())
#===================================================================================================================
# [Running Random Forest Model]
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =10, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)
# Make predictions for Random Forrest Algorithm
prediction_test = model_rf.predict(X_test)
print("Random Forest Classifier Prediction Score")
RandomForest = metrics.accuracy_score(y_test, prediction_test)
print(metrics.accuracy_score(y_test, prediction_test))
dict['Random Forest'] = RandomForest
#
#Code for Splitting into four Quadrants
#0-0 True Positive - Actual and Predicted values are Correct
#0-1 False Negative - Actual Values are Correct and Predicted values are wrong
#1-0 False Positive - Actual Values are Wrong and Predicted values are correct 
#1-1 True Negative - Actual Values and Predictive values are Incorrect
print ("Confusion Matrix for Random Forrest")
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,prediction_test)
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True)
plt.show()
#If user wanted to print the Classification Report, Uncomment the below 2 lines of Code and Execute it
#from sklearn.metrics import classification_report
#print(classification_report(y_test,prediction_test))
# This code is meant for printing the Attributes with highest weightage / Prediction
print ("Attributes affecting the Customer to Port Out from Existing Service Provider")
importances = model_rf.feature_importances_
weights = pd.Series(importances,index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')
#===================================================================================================================
#Deep Learning Algo


print('All Alogorithm with there score ',dict)
max_algo = max(dict, key = dict.get)
print('The Algorithm with Highest predection value is ',max_algo)