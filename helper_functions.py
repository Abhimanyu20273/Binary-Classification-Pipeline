import numpy as np
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def accuracy_precison_recall_function_for_all(test_y,sum_of_prob,predicted_label,type_,n=0,threshold=0):
  cf_mat = confusion_matrix(test_y, predicted_label)
  true_negative = cf_mat[0][0]
  false_positive = cf_mat[0][1]
  false_negative = cf_mat[1][0]
  true_positive = cf_mat[1][1]
  accuracy = (true_positive + true_negative)/(true_negative + false_positive + false_negative + true_positive)
  precision = true_positive/(false_positive + true_positive)
  sensitivity = true_positive/(false_negative + true_positive)
  specificity = true_negative/(false_positive+true_negative)
  f1_score_val = (2 * sensitivity * precision)/(sensitivity + precision)
  mcc = matthews_corrcoef(test_y, predicted_label)
  if(type_ == "simple"):
    auroc = roc_auc_score(test_y, sum_of_prob)
    return [auroc,accuracy,precision,sensitivity,specificity,f1_score_val,mcc]
  if(type_ == "Without_AUROC"):
    return [n,threshold,accuracy,precision,sensitivity,specificity,f1_score_val,mcc]
  else:
    auroc = roc_auc_score(test_y, sum_of_prob)
    return [n,threshold,auroc,accuracy,precision,sensitivity,specificity,f1_score_val,mcc]
def senstivity_specificity_model(threshold_arr,type_):
  min_val = 1
  min_val_index = -1
  if(type_ == "simple"):
    for t in range(len(threshold_arr)):
      if((abs(threshold_arr[t][4] - threshold_arr[t][3])) < min_val):
        min_val = abs(threshold_arr[t][4] - threshold_arr[t][3])
        min_val_index = t
  else:
    for t in range(len(threshold_arr)):
      if(abs(threshold_arr[t][6] - threshold_arr[t][5]) < min_val):
        min_val = abs(threshold_arr[t][6] - threshold_arr[t][5])
        min_val_index = t
  return threshold_arr[min_val_index]
def model_building_individual(num_features,train_x,train_y,model_type,parameter_optimization):
  models = []
  if(model_type == "LR"):
    for i in range(0,num_features):
      if(parameter_optimization == "yes"):
        LR_model = LogisticRegression(class_weight='balanced')
        param_grid = [{'C':[0.001,0.01,0.1,1,10,100],'penalty':['l1','l2'],'solver':['liblinear']}]
        model_gscv = GridSearchCV(estimator = LR_model,param_grid = param_grid,scoring = 'roc_auc',refit=True).fit(train_x[:,i].reshape(train_x.shape[0],1),train_y)  
        models.append(model_gscv.best_estimator_)
      else:
        models.append(LogisticRegressionCV(max_iter=1000,class_weight='balanced').fit(train_x[:,i].reshape(train_x.shape[0],1),train_y))
  if(model_type == "EN"):
    for i in range(0,num_features):
      models.append(LogisticRegressionCV(max_iter=1000,class_weight='balanced',cv = 5,penalty='elasticnet',solver = 'saga',l1_ratios=[0.5, 0.5, 0.5,0.5,0.5]).fit(train_x[:,i].reshape(train_x.shape[0],1),train_y))
  if(model_type == "NB"):
    for i in range(0,num_features):
      models.append(GaussianNB().fit(train_x[:,i].reshape(train_x.shape[0],1),train_y))
  if(model_type == "SVM"):
    for i in range(0,num_features):
      if(parameter_optimization == "yes"):
        svm_model = SVC(class_weight='balanced')
        param_grid = [{'C':[0.001,0.01,0.1,1,10,100],'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}]
        model_gscv = GridSearchCV(estimator = svm_model,param_grid = param_grid,scoring = 'roc_auc',refit=True).fit(train_x[:,i].reshape(train_x.shape[0],1),train_y)  
        models.append(model_gscv.best_estimator_)
      else:
        models.append(SVC(class_weight='balanced').fit(train_x[:,i].reshape(train_x.shape[0],1),train_y))
  if(model_type == "RF"):
    for i in range(0,num_features):
      if(parameter_optimization == "yes"):
        rf_model = RandomForestClassifier(class_weight='balanced')
        param_grid = [{'criterion':["gini", "entropy"],'max_depth':[6,12,18],"n_estimators":[100,150]}]
        model_gscv = GridSearchCV(estimator = rf_model,param_grid = param_grid,scoring = 'roc_auc',refit=True).fit(train_x[:,i].reshape(train_x.shape[0],1),train_y)  
        models.append(model_gscv.best_estimator_)
      else:
        models.append(RandomForestClassifier(class_weight='balanced').fit(train_x[:,i].reshape(train_x.shape[0],1),train_y))
  return models

