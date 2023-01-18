from helper_functions import model_building_individual
import pandas as pd
import numpy as np
from scipy import stats
import argparse

#Function to build separate machine learning models for each feature.


#Taking command line arguments and processing them
help_message = "Program to featurerate a CSV file with probability of the sample being positive instead of the feature expression value"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Train_csv",default = "Scaled_train_data.csv",help="The train data file")
cmd_argument_parser.add_argument("Test_csv",default = "Scaled_test_data.csv", help="The test data file")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("Model",default = "LR", help="Model to be used. Enter LR for Logistic Regression,EN for elastinet,NB for gaussian naive bayes,SVM for support vector machine and RF for random forest")
cmd_argument_parser.add_argument("parameter_optimization",default = "yes", help="Enter yes for paramater optimization using grid search else enter no")
cmd_arguments = cmd_argument_parser.parse_args()

train_dataframe = pd.read_csv(cmd_arguments.Train_csv)
test_dataframe = pd.read_csv(cmd_arguments.Test_csv)
label_column = cmd_arguments.label_column
parameter_optimization = cmd_arguments.parameter_optimization
label_list = train_dataframe[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]


#Splitting the data into train,test and also feature expressions,labels
train_x = np.array((train_dataframe.loc[:,train_dataframe.columns != label_column]))
train_y = (np.array(train_dataframe.loc[:,train_dataframe.columns == label_column]))
train_y = train_y.reshape(len(train_y))
test_x = np.array((test_dataframe.loc[:,test_dataframe.columns != label_column]))
test_y = np.array(test_dataframe.loc[:,test_dataframe.columns == label_column])
test_y = test_y.reshape(len(test_y))
# print(label1)
# print(label2)
for i in range(len(test_y)):
	if(test_y[i] == label1):
		test_y[i] = -1
	elif(test_y[i] == label2):
		test_y[i] = 1
test_y = test_y.astype(np.int64)
for i in range(len(train_y)):
	if(train_y[i] == label1):
		train_y[i] = -1
	elif(train_y[i] == label2):
		train_y[i] = 1
train_y = train_y.astype(np.int64)
# print(train_y.shape)
# print(test_y.shape)
models = model_building_individual(train_x.shape[1],train_x,train_y,cmd_arguments.Model,parameter_optimization)
#Creating another array which has probability value of the sample being positive for each feature_expression_value_of_each_sample 
probability_arr_train = []
probability_arr_test = []
for i in range(train_x.shape[1]):
	probability_arr_train.append(models[i].predict_proba(train_x[:,i].reshape(train_x.shape[0],1))[:,1])
	probability_arr_test.append(models[i].predict_proba(test_x[:,i].reshape(test_x.shape[0],1))[:,1])

#Writing the result into two csv files 
probability_arr_train = np.array(probability_arr_train)
probability_arr_test = np.array(probability_arr_test)
probability_arr_train = np.transpose(probability_arr_train)
probability_arr_test = np.transpose(probability_arr_test)
train_data = np.append(probability_arr_train, train_y.reshape(len(train_y),1), axis=1)
test_data = np.append(probability_arr_test, test_y.reshape(len(test_y),1), axis=1)
feature_names = train_dataframe.columns
train_dataframe = pd.DataFrame(train_data)
train_dataframe.columns = feature_names  
train_dataframe.to_csv('Train_data_probability.csv',index = False)
test_dataframe = pd.DataFrame(test_data) 
test_dataframe.columns = feature_names 
test_dataframe.to_csv('Test_data_probability.csv',index = False)
