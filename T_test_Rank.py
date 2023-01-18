import pandas as pd
import numpy as np
from scipy import stats
from math import isnan
import argparse
 
# program to rank features on basis of t-test

#Taking command line arguments and processing them
help_message = "Program to rank features on basis of t-test(p-value)"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Train_csv", help="The train data file")
cmd_argument_parser.add_argument("Test_csv", help="The test data file")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("threshold_",default = 0 ,help="Threshold for feature removal based on p-value. By default 0.05.")
cmd_arguments = cmd_argument_parser.parse_args()
whole_df = pd.read_csv(cmd_arguments.Train_csv)
label_column = cmd_arguments.label_column
threshold_ = float(cmd_arguments.threshold_)
feature_list = list(whole_df.columns)
test_dataframe = pd.read_csv(cmd_arguments.Test_csv)
feature_list.remove(label_column)
label_list = whole_df[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]
#Preparing labelwise data
column_names = []
count = 0;
for col in whole_df.columns:
	if(col !=label_column):
		column_names.append(col)
	count+=1
label_1_dataframe = (whole_df.loc[whole_df[label_column] == label1])
label_2_dataframe = (whole_df.loc[whole_df[label_column] == label2])
label_1_dataframe.pop(label_column)
label_2_dataframe.pop(label_column)
label_1_dataframe = label_1_dataframe.astype(float)
label_2_dataframe = label_2_dataframe.astype(float)

label_1_data = np.array(label_1_dataframe)
label_2_data = np.array(label_2_dataframe)
#T-test
t_test_result = stats.ttest_ind(label_1_data, label_2_data,equal_var = False,nan_policy='propagate')
result_arr = []
feature_positions_dict = {}
position = 0
for i in range(0,label_1_data.shape[1]):
	if(isnan(t_test_result[1][i]) == False):
		result_arr.append([position,t_test_result[0][i],t_test_result[1][i]])
		feature_positions_dict[position] = column_names[i]  
		position +=1
result_arr.sort(key=lambda result_arr:result_arr[2])
for i in range(0,position):
	result_arr[i][0] = feature_positions_dict[result_arr[i][0]]
result_arr = np.array(result_arr)
dataframe = pd.DataFrame(result_arr) 
dataframe.columns = ["Feature name","Calculated t-statistic","P value"]
convert_col = {'Calculated t-statistic': float,'P value': float}
dataframe = dataframe.astype(convert_col)
dataframe.to_csv('T_Test.csv',index = False)

dataframe = dataframe[dataframe['P value']<threshold_].copy()
feature_names_with_high_p_value = dataframe.loc[:,"Feature name"]
feature_names = np.array(feature_names_with_high_p_value)
feature_names = np.append(feature_names, label_column)
whole_df = whole_df.loc[:,feature_names]
whole_df.to_csv(cmd_arguments.Train_csv,index = False)
test_dataframe = test_dataframe.loc[:,feature_names]
test_dataframe.to_csv(cmd_arguments.Test_csv,index = False)