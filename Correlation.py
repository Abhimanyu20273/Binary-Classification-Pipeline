from helper_functions import accuracy_precison_recall_function_for_all,senstivity_specificity_model,model_building_individual
import pandas as pd
import numpy as np
import argparse

help_message = "Program to combine the top n number of features. Running Individual_feature.py before this program is a mandatory condition for it to work"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("Scaled_train_data_csv", help="The scaled train data file")
cmd_argument_parser.add_argument("Scaled_test_data_csv", help="The scaled test data file")
cmd_argument_parser.add_argument("individual_feature_file", help="Individual feature file result")
cmd_argument_parser.add_argument("train_data_correlation", help="train_data_unscaled")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("threshold_",default = 0.8, help="Threshold for removal of features based on correlation. By default its 0.8.")
cmd_arguments = cmd_argument_parser.parse_args()

train_data_csv = cmd_arguments.Scaled_train_data_csv
test_data_csv = cmd_arguments.Scaled_test_data_csv
individual_feature_csv = cmd_arguments.individual_feature_file
train_data_correlation = cmd_arguments.train_data_correlation
threshold_ = float(cmd_arguments.threshold_)
train_dataframe = pd.read_csv(train_data_csv)
test_dataframe = pd.read_csv(test_data_csv)
top_feature_dataframe = pd.read_csv(individual_feature_csv)
label_column = cmd_arguments.label_column
label_list = train_dataframe[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]
train_data_corr_dataframe = pd.read_csv(train_data_correlation)
train_data_corr_columns = train_data_corr_dataframe.columns
req_columns  = np.array(top_feature_dataframe.loc[:,"feature name"])
for i in range(len(train_data_corr_columns)):
	if(train_data_corr_columns[i] not in req_columns):
		train_data_corr_dataframe.drop(train_data_corr_columns[i], inplace=True, axis=1) 

correlation = train_data_corr_dataframe.corr()
unstacked_dataframe = correlation.abs().unstack()
sorted_data = unstacked_dataframe.sort_values()
indices = sorted_data.index
remove_label = []
already_seen_dict = {}
for i in range(len(sorted_data)):
	if(sorted_data[i] > threshold_ and indices[i][0] !=indices[i][1] and (indices[i][0] + indices[i][1]) not in already_seen_dict):
		already_seen_dict[indices[i][0]+indices[i][1]] = 0
		already_seen_dict[indices[i][1]+indices[i][0]] = 0
		if(((top_feature_dataframe.loc[top_feature_dataframe["feature name"] == indices[i][1]])["AUROC"].iat[0] > (top_feature_dataframe.loc[top_feature_dataframe["feature name"] == indices[i][0]])["AUROC"].iat[0])):
			remove_label.append(indices[i][0])
		else:
			remove_label.append(indices[i][1])
remove_label = np.unique(remove_label)
#Dropping columns
for i in range(len(remove_label)):
	train_dataframe.drop(remove_label[i], inplace=True, axis=1)
	test_dataframe.drop(remove_label[i], inplace=True, axis=1)
	top_feature_dataframe.drop(top_feature_dataframe[top_feature_dataframe["feature name"] == remove_label[i]].index, inplace = True)

new_feature_list = np.array(top_feature_dataframe["feature name"])
new_feature_list = np.append(new_feature_list, label_column)
train_dataframe = train_dataframe.loc[:,new_feature_list]
train_dataframe.to_csv(cmd_arguments.Scaled_train_data_csv,index = False)
test_dataframe = test_dataframe.loc[:,new_feature_list]
test_dataframe.to_csv(cmd_arguments.Scaled_test_data_csv,index = False)