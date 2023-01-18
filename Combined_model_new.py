from helper_functions import accuracy_precison_recall_function_for_all,senstivity_specificity_model,model_building_individual
import pandas as pd
import numpy as np
from scipy import stats
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def accuracy_with_num_features_model_prob(test_x_prob,test_y,num_features):
	threshold_arr = []
	predicted_label = []
	for l in range(20,80):
		for i in range(0,test_x_prob.shape[0]):
			if(test_x_prob[i]<=l/100):
				y = -1
			else:
				y = 1
			predicted_label.append(y)
		threshold_arr.append(accuracy_precison_recall_function_for_all(test_y,test_x_prob,predicted_label,"Other",num_features,l/100))
		predicted_label = []
	return senstivity_specificity_model(threshold_arr,"Other")

def accuracy_with_num_features_model_prob_validate(test_x_prob,test_y,threshold,num_features):
	num_features_accuracy_test_complete = []
	threshold_arr = []
	predicted_label = []
	for i in range(0,test_x_prob.shape[0]):
		if(test_x_prob[i] <=threshold):
			y = -1
		else:
			y = 1
		predicted_label.append(y)
	num_features_accuracy_test_complete.append(accuracy_precison_recall_function_for_all(test_y,test_x_prob,predicted_label,"Other",num_features,threshold))
	return num_features_accuracy_test_complete

help_message = "Program to combine the top n number of features. Running Individual_feature.py before this program is a mandatory condition for it to work"
cmd_argument_parser = argparse.ArgumentParser(description = help_message)
cmd_argument_parser.add_argument("train_data_csv", help="The training data file")
cmd_argument_parser.add_argument("test_data_csv", help="The test data file")
cmd_argument_parser.add_argument("individual_feature_file", help="Individual feature file")
cmd_argument_parser.add_argument("train_data_correlation", help="train_data_unscaled")
cmd_argument_parser.add_argument("remove_features_correlation", help="Enter 'Y' to remove features on basis of correlation. Any other character otherwise. ")
cmd_argument_parser.add_argument("label_column", help="Column name which has the labels")
cmd_argument_parser.add_argument("Model",default = "LR", help="Model to be used. Enter LR for Logistic Regression,EN for elastinet,NB for gaussian naive bayes,SVM for support vector machine and RF for random forest")
cmd_argument_parser.add_argument("num_top_features_",default = "20", help="Number of top features to be used")
cmd_argument_parser.add_argument("output_file", help="Name of output file")

cmd_arguments = cmd_argument_parser.parse_args()

train_data_csv = cmd_arguments.train_data_csv
model_type = cmd_arguments.Model
remove_features_correlation = cmd_arguments.remove_features_correlation
test_data_csv = cmd_arguments.test_data_csv
individual_feature_csv = cmd_arguments.individual_feature_file
train_data_correlation = cmd_arguments.train_data_correlation
top_feature_dataframe = pd.read_csv(individual_feature_csv)
num_top_features_ = int(cmd_arguments.num_top_features_)

train_dataframe = pd.read_csv(train_data_csv)
test_dataframe = pd.read_csv(test_data_csv)
label_column = cmd_arguments.label_column
label_list = train_dataframe[label_column].unique()
label1 = label_list[0]
label2 = label_list[1]
output_file = cmd_arguments.output_file

if(remove_features_correlation == "Y"):
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
		if(sorted_data[i] > 0.8 and indices[i][0] !=indices[i][1] and (indices[i][0] + indices[i][1]) not in already_seen_dict):
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
num_top_features = min(num_top_features_,top_feature_dataframe.shape[0])
top_feature_dataframe = top_feature_dataframe.head(num_top_features)

top_feature_list = np.array(top_feature_dataframe["feature name"])

feature_list = list(train_dataframe.columns)
feature_list.remove(label_column)

train_x = np.array((train_dataframe.loc[:,top_feature_list]))
train_y = (np.array(train_dataframe.loc[:,train_dataframe.columns == label_column]))
train_y = train_y.reshape(len(train_y))
validation_x = np.array(test_dataframe.loc[:,top_feature_list])
validation_y = np.array(test_dataframe.loc[:,test_dataframe.columns == label_column])
validation_y = validation_y.reshape(len(validation_y))

num_features_accuracy_test_arr = []

for i in range(validation_y.shape[0]):
  if(validation_y[i] == label1):
    validation_y[i] = -1
  if(validation_y[i] == label2):
    validation_y[i] = 1
validation_y = validation_y.reshape(len(validation_y))
validation_y = validation_y.astype("int64")

for i in range(train_y.shape[0]):
  if(train_y[i] == label1):
    train_y[i] = -1
  if(train_y[i] == label2):
    train_y[i] = 1
train_y = train_y.reshape(len(train_y))
train_y = train_y.astype("int64")

train_data__x_fold, test_data_x_fold, train_data_y_fold, test_data_y_fold = train_test_split(train_x, train_y, test_size=0.20,stratify=train_y,random_state=42)
best_metrics = None
accuracy_metric_list = []
mymodel = None
for i in range(2,num_top_features):
	if(model_type == "LR"):
		mymodel = LogisticRegression(class_weight='balanced')
	if(model_type == "NB"):
		mymodel = GaussianNB()
	if(model_type == "EN"):
		mymodel = LogisticRegressionCV(max_iter=1000,class_weight='balanced',cv = 5,penalty='elasticnet',solver = 'saga',l1_ratios=[0.5, 0.5, 0.5,0.5,0.5])
	if(model_type == "RF"):
		mymodel = RandomForestClassifier(class_weight='balanced')
	if(model_type == "SVM"):
		mymodel = SVC(class_weight='balanced')
	sfs_obj = SequentialFeatureSelector(mymodel, n_features_to_select=i,scoring = 'roc_auc')
	sfs_obj.fit(train_data__x_fold, train_data_y_fold)
	train_fitted_x = sfs_obj.transform(train_data__x_fold)
	mymodel.fit(train_fitted_x,train_data_y_fold)
	test_fitted_x = sfs_obj.transform(test_data_x_fold)
	accuracy_metric = accuracy_with_num_features_model_prob(mymodel.predict_proba(test_fitted_x)[:,1],test_data_y_fold,i)
	accuracy_metric_list.append(accuracy_metric)
	if(best_metrics == None):
		best_metrics = accuracy_metric
	elif(best_metrics[2] < accuracy_metric[2]):
		best_metrics = accuracy_metric

accuracy_metric_ = np.array(accuracy_metric_list)
num_features_accuracy_test_dataframe = pd.DataFrame(accuracy_metric_)
num_features_accuracy_test_dataframe.columns = ["Number of features train", "Train Threshold","Train AUROC","Train Accuracy","Train Precision","Train Sensitivity","Train Specificity","Train F1_score","Train MCC"]
num_features_accuracy_test_dataframe.to_csv("Train" + output_file)


if(model_type == "LR"):
	mymodel = LogisticRegression(class_weight='balanced')
if(model_type == "NB"):
	mymodel = GaussianNB()
if(model_type == "EN"):
	mymodel = LogisticRegressionCV(max_iter=1000,class_weight='balanced',cv = 5,penalty='elasticnet',solver = 'saga',l1_ratios=[0.5, 0.5, 0.5,0.5,0.5])
if(model_type == "RF"):
	mymodel = RandomForestClassifier(class_weight='balanced')
if(model_type == "SVM"):
	mymodel = SVC(class_weight='balanced')
sfs_obj = SequentialFeatureSelector(mymodel, n_features_to_select=best_metrics[0],scoring = 'roc_auc')
sfs_obj.fit(train_x, train_y)
train_fitted_x = sfs_obj.transform(train_x)
mymodel.fit(train_fitted_x,train_y)
validation_fitted_x = sfs_obj.transform(validation_x)

num_features_accuracy_validate = accuracy_with_num_features_model_prob_validate(mymodel.predict_proba(validation_fitted_x)[:,1],validation_y,best_metrics[1],best_metrics[0])
num_features_accuracy_validate = np.array(num_features_accuracy_validate)
num_features_accuracy_validate_reshaped = num_features_accuracy_validate.reshape(1,9)
num_features_accuracy_validatation_dataframe = pd.DataFrame(num_features_accuracy_validate_reshaped)
num_features_accuracy_validatation_dataframe.columns = ["Number of features val", "Validation Threshold","Validation AUROC","Validation Accuracy","Validation Precision","Validation Sensitivity","Validation Specificity","Validation F1_score","Validation MCC"]
num_features_accuracy_validatation_dataframe.to_csv("Test" + output_file)