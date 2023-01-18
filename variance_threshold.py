import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from sklearn.feature_selection import VarianceThreshold
#Taking command line arguments and processing them
help_message = "Program to remove features that have a lower variance than a specified threshold. Default is 0"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Train_csv", help="The train data file")
cmd_argument_parser.add_argument("Test_csv", help="The test data file")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("threshold_",default = 0 ,help="Threshold for variance removal")
cmd_arguments = cmd_argument_parser.parse_args()
train_dataframe = pd.read_csv(cmd_arguments.Train_csv)
test_dataframe = pd.read_csv(cmd_arguments.Test_csv)
label_column = cmd_arguments.label_column
threshold_ = float(cmd_arguments.threshold_)
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
	
train_x = np.array((train_dataframe.loc[:,column_names]))
train_y = (np.array(train_dataframe.loc[:,train_dataframe.columns == label_column]))
test_x = np.array((test_dataframe.loc[:,column_names]))
test_y = np.array(test_dataframe.loc[:,test_dataframe.columns == label_column])
normalised_train = train_x/(train_x.mean())
variance_finder = VarianceThreshold(threshold = threshold_).fit(train_x)
req_features = variance_finder.get_feature_names_out(input_features = column_names)
req_features = np.append(req_features, label_column)

train_dataframe = train_dataframe.loc[:,req_features]
test_dataframe = test_dataframe.loc[:,req_features]
train_dataframe.to_csv(cmd_arguments.Train_csv,index = False)
test_dataframe.to_csv(cmd_arguments.Test_csv,index = False)



