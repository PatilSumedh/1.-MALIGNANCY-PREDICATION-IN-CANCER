import numpy as np
import sklearn.datasets #importing datasets for breast cancer

import pandas as pd

#importing logistic regression
from sklearn.linear_model import LogisticRegression

#import accurracy score for evaluation of model
from sklearn.metrics import accuracy_score

#accessing the dataset
b_cancer = sklearn.datasets.load_breast_cancer()

X = b_cancer.data       #X--> represents training data
Y = b_cancer.target     #Y--> represents labels

#importing data to panda dataframe
data = pd.DataFrame(b_cancer.data,columns = b_cancer.feature_names)

data['class'] = b_cancer.target

print(b_cancer.target_names)  #prints target elements i.e malignant and benign

#training and testing the model using the train_test_split

X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.1 ,stratify = Y,random_state = 1)
#test_size    --> to speciffy the percentage of the test data needed
#stratify     --> for correct distribution of the data as of the original data
#random_state --> specific split of data . Each value of random_state splits the data differently

#Loading logistic regression model to clf
clf = LogisticRegression()

#training the model begins here
clf.fit(X_train,Y_train)

pred_tndata = clf.predict(X_train)                   #prediction on training data
tn_accuracy = accuracy_score(Y_train,pred_tndata)    #checking accuracy of training data
print("Accuracy on training data :",tn_accuracy)     #printing accuracy of training data

pred_tsdata =clf.predict(X_test)                     #prediction on testing data
ts_accuracy = accuracy_score(Y_test,pred_tsdata)     #checking accuracy of testing data
print("Accuracy on testing data :",ts_accuracy)      #printing accuracy of testing data

#Detecting stage of cancer i.e benign or malignant

input_data = tuple(map(float,input("Enter Data:").split(',')))

#change input data to array from tuple for making prediction
input_arr = np.asarray(input_data)

#reshape the array as we are predicting the output for one instance
reshape_data = input_arr.reshape(1,-1)

#final prediction 
prediction = clf.predict(reshape_data) 

print(prediction)  #gives output as [0] or [1] i.e malignant or begnin respectively.

#returns a list with [0] if malignant or [1] if benign

if prediction[0]==0:
    print("The breast cancer is Malignant")
else:
    print("The breast cancer is Benign")
    

#Sample data 1 (M) : 17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189

#Sample data 2 (B) : 13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259

#Sample data 3 (M) : 20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902

#Sample data 4 (M) : 19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758

#Sample data 5 (B) : 13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183
