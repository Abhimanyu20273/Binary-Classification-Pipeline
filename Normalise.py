import pandas as pd
import numpy as np
from scipy import stats
import argparse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
 
#Taking command line arguments and processing them
help_message = "Program to  normalise data"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Train_csv",help="The train data file")
cmd_argument_parser.add_argument("Test_csv", help="The test data file")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("Scaler_type", default = "ss",help="Enter ss for standard scaler and mms for min max scaler")
cmd_arguments = cmd_argument_parser.parse_args()
train_dataframe = pd.read_csv(cmd_arguments.Train_csv)
test_dataframe = pd.read_csv(cmd_arguments.Test_csv)
label_column = cmd_arguments.label_column
Scaler_type = cmd_arguments.Scaler_type
label_list = train_dataframe[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]

#Separating labels and feature exprssion data. Converting labels to 1 or -1   
column_list = []
column_names = []
count = 0;
for col in train_dataframe.columns:
	if(col != label_column):
		column_list.append(count)
		column_names.append(col)
	count+=1

#scale the data
train_x = np.array((train_dataframe.loc[:,column_names]))
train_y = (np.array(train_dataframe.loc[:,train_dataframe.columns == label_column]))
test_x = np.array((test_dataframe.loc[:,column_names]))
test_y = np.array(test_dataframe.loc[:,test_dataframe.columns == label_column])
if(Scaler_type == "mms"):
	scaler  = preprocessing.StandardScaler().fit(train_x)
else:
	scaler  = preprocessing.MinMaxScaler().fit(train_x)

train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
train_data = np.append(train_x_scaled, train_y, axis=1)
test_data = np.append(test_x_scaled, test_y, axis=1)

#Write the result into two csv files
column_names.append(label_column)
train_dataframe = pd.DataFrame(train_data)
train_dataframe.columns = column_names  
train_dataframe.to_csv('Scaled_train_data.csv',index = False)
test_dataframe = pd.DataFrame(test_data) 
test_dataframe.columns = column_names 
test_dataframe.to_csv('Scaled_test_data.csv',index = False)

