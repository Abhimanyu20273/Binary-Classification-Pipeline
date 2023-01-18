import argparse
import sys

import numpy as np
import pandas as pd
from helper_functions import (accuracy_precison_recall_function_for_all,
                              model_building_individual,
                              senstivity_specificity_model)
from scipy import stats
from sklearn.model_selection import StratifiedKFold

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def feature_wise_accuracy_prob(test_x,feature_list,test_y):
  predicted_label = []
  arr_for_csv = []
  single_feature_arr = []
  num_features = test_x.shape[1]
  for j in range(0,num_features):
    for l in range(20,80):
      for i in range(0,test_x.shape[0]):
        y = test_x[i][j]
        if(y <=(l/100)):
          y = -1
        else:
          y = 1
        predicted_label.append(y)
      single_feature_arr.append(accuracy_precison_recall_function_for_all(test_y,test_x[:,j].reshape(test_x.shape[0],1),predicted_label,"Other",feature_list[j],l/100))
      predicted_label = []
    max_val = 0
    max_val_index = -1
    for t in range(len(single_feature_arr)):
      if(min(single_feature_arr[t][6],single_feature_arr[t][5]) > max_val):
        max_val = min(single_feature_arr[t][6],single_feature_arr[t][5])
        max_val_index = t
    arr_for_csv.append(senstivity_specificity_model(single_feature_arr,"other"))
    single_feature_arr = []
  return arr_for_csv
def feature_wise_accuracy_prob_validate(test_x,feature_list,test_y,individual_feature_arr):
  predicted_label = []
  arr_for_csv = []
  single_feature_arr = []
  num_features = test_x.shape[1]
  for j in range(0,num_features):
    threshold = individual_feature_arr[j][0]
    for i in range(0,test_x.shape[0]):
      y = test_x[i][j]
      if(y <= threshold):
        y = -1
      else:
        y = 1
      predicted_label.append(y)
    single_feature_arr.append(accuracy_precison_recall_function_for_all(test_y,test_x[:,j].reshape(test_x.shape[0],1),predicted_label,"Other",feature_list[j],threshold))
    predicted_label = []
    arr_for_csv.append(single_feature_arr[0])
    single_feature_arr = []
  return arr_for_csv

help_message = "Program to calculate auroc,accuracy,precision,f1_score,sensitivity for individual features and sort the result according to auroc"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("train_data_csv", help="The training data file")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("output_file", help="Name of output file")
cmd_arguments = cmd_argument_parser.parse_args()
train_data_csv = cmd_arguments.train_data_csv
train_dataframe = pd.read_csv(train_data_csv)
label_column = cmd_arguments.label_column
output_file = cmd_arguments.output_file
label_list = train_dataframe[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]
feature_list = list(train_dataframe.columns)
feature_list.remove(label_column)
column_list = []
column_names = []
count = 0
for col in train_dataframe.columns:
  if(col != label_column):
    column_list.append(count)
    column_names.append(col)


#Splitting the data into feature expressions,labels
train_x = np.array((train_dataframe.loc[:,column_names]))
train_y = (np.array(train_dataframe.loc[:,train_dataframe.columns == label_column]))
result_arr = []
models = []
train_y = train_y.reshape(len(train_y))
for i in range(train_y.shape[0]):
  if(train_y[i] == label1):
    train_y[i] = -1
  if(train_y[i] == label2):
    train_y[i] = 1
train_y = train_y.astype("int64")
folds_genarator = StratifiedKFold()
i =0
for train_index, test_index in folds_genarator.split(train_x, train_y):
  train_data_x_5_fold = train_x[train_index]
  train_data_x_5_fold = train_data_x_5_fold.astype(np.float64)
  train_data_y_5_fold = train_x[train_index]
  train_data_y_5_fold = train_data_y_5_fold.astype(np.int64)

  test_data_x_5_fold = train_x[test_index]
  test_data_x_5_fold = test_data_x_5_fold.astype(np.float64)
  test_data_y_5_fold = train_y[test_index]
  test_data_y_5_fold = test_data_y_5_fold.astype(np.int64)
  result = feature_wise_accuracy_prob(test_data_x_5_fold, feature_list, test_data_y_5_fold)
  if(i ==0):
    result_arr = result
  else:
    result_arr = np.concatenate((result_arr,result))
  i+=1


#Writing the result into a csv file 
dataframe = pd.DataFrame(result_arr)
dataframe.columns = ["feature name","Threshold","AUROC","Accuracy","Precision","Sensitivity","Specificity","F1_score","MCC"]
convert_col = {"Threshold" : float ,'AUROC': float,'Accuracy': float,'Precision': float,'Sensitivity': float,'Specificity': float,'F1_score' : float,"MCC" : float}
dataframe = dataframe.astype(convert_col)
dataframe = dataframe.groupby([dataframe.columns[0]]).mean()
individual_feature_arr = np.array(dataframe)
dataframe = dataframe.sort_values(by=['AUROC'], ascending=False)
dataframe.to_csv(output_file)



