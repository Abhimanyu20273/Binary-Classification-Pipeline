import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
help_message = "Program to split the data into test and train"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Inp_csv", help="The csv file which has the feature expression values")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("stratify",default = "yes", help="Whether to use stratified train test split. By default program uses stratified train test split, enter no to use non-stratifed train test split")
cmd_arguments = cmd_argument_parser.parse_args()
whole_df = pd.read_csv(cmd_arguments.Inp_csv)
label_column = cmd_arguments.label_column
stratify = cmd_arguments.stratify
column_list = []
column_names = []
count = 0;
for col in whole_df.columns:
	if(col != label_column):
		column_list.append(count)
		column_names.append(col)
	count+=1
data_x = np.array((whole_df.iloc[:,column_list]))
data_y = np.array(whole_df.loc[:,whole_df.columns == label_column])
train_x = []
train_y = []
test_x = []
test_y = []
if(stratify == "yes"):
	train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20,stratify=data_y,random_state=42)
else:
	train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20,stratify=data_y,random_state=42)
train_data = np.append(train_x, train_y, axis=1)
test_data = np.append(test_x, test_y, axis=1)
column_names.append(label_column)
train_dataframe = pd.DataFrame(train_data)
train_dataframe.columns = column_names  
train_dataframe.to_csv('Train_data.csv',index = False)
test_dataframe = pd.DataFrame(test_data) 
test_dataframe.columns = column_names 
test_dataframe.to_csv('Test_data.csv',index = False)
