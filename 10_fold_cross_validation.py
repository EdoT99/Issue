import os
import numpy as np
import pandas as pd
import glob
from itertools import cycle
import random
import numbers
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statistics import mean
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

try:
  from imblearn.over_sampling import SMOTE

except ModuleNotFoundError:
  print("module 'SMOTE' is not installed")

#this function, given a dataset, provides number of attributes(columns) and classes
def attr_count(df):
    count = 0
    for col in df.columns:
        if col == 'Species' or col == 'type' or col == 'Class' or col =='Id' or col == 'class':
            classe = col

            continue
        else: 
            count += 1

    return count, df[classe].value_counts()


def dataset_parser(df,attributes,cumida):
  if cumida == True:
    df = df.iloc[: , 1:] 
  else: 
    df = df.iloc[:, 0:] 
  classes = df.iloc[:,0].unique()
  encode_class = [[clas,i] for i,clas in enumerate(classes)]
  print(df.columns)
  df.rename(columns={df.columns[0]: 'Class'}, inplace=True)
  print(encode_class)
  # transforming class names to binary
  for classe in encode_class:
    df.loc[df['Class'] == str(classe[0]),'Class'] = str(classe[1])
  labels = df['Class'].astype('int')
  #only attributes (variables/ genes)

  if attributes == True:               #if user does not specify number of attributes, takes all the columns
    vars = [ var for var in df.columns]        
    data = df[vars[1:]].values
    print(len(vars))
  else:                                  #if users sets it, takes only the number of attributes typed
    vars = [ var for var in df.columns[:attributes+1]]  
    data = df[vars[1:]].values

  labels = df[vars[0]].astype('int')

  return data,labels, vars

#This function, search the MAX occurring class, for each minority class it calculates the N of examples to be oversampled,
#Returns a list of tuples: key class, integer examples
def max_occurrence(Y):
    lista = []
    counts = Counter(Y)
    max_occurring= max(counts, key=counts.get)
    max_value = counts[max_occurring]
    print('Max_occurring class:',max_occurring)
    for key,value in counts.items():
        if key != max_occurring:
            val = max_value - value
            lista.append((key,val))
    return lista

#this function  appends newly generated rows (examples) by KDE to existing original ones;
#Returns X data anD Y labels with new examples oversampled by KDE
def augmenting_df(list_a,X,Y):
  print('Raw training: ',X.shape)
  new_arrays, new_labels = [], []

  for clas in list_a:  #iterating over class symbol and array
      array = clas[1]
      labels = [ clas[0] for i in range(len(array))]
      new_labels.append(labels)
      new_arrays.append(array) 

  print('--------------------------------------------------------')
  
  labels = np.concatenate(new_labels,axis= 0)
  #print(labels)
  arrays = np.concatenate(new_arrays, axis = 0)
  #print(arrays)

  print(arrays.shape)
  #arrays = np.concatenate(arrays,axis = 0)
  print(array.shape)
  new_rows = np.vstack([labels,arrays.T]).T
  #print(new_rows)
  original_rows = np.vstack([Y,X.T]).T
  
#appending arryays is generally faster than stacking, do not stack while iterating, it increase memory usage
  oversamp = np.vstack([original_rows,new_rows])
  print(oversamp)
  print('KDE final oversampled (classes + data): ', oversamp.shape)
  X_kde = oversamp[:,1:]
  Y_kde = oversamp[:,0]
  #print(Y_kde)
  #print(X_kde[:,1:])
  
  return  X_kde,Y_kde


#oversampling minority class (all the minority if dataset multiclass) 
def oversamp_KDE_definitive(X,Y,kernel):
    
    list_samples = []
    #call function to provide minority classes and missing examples for each to match majoirty class
    lista = max_occurrence(Y)
    #for every class in dataframe, oversamples a number of istances to match maj class
    for item in lista:
        
        classe, n_istances = item[0], item[1]
      #selecting minority examples BY INDEX
        
        indices = [i for i, class_value in enumerate(Y) if class_value == classe]
        data = X[indices,:]
        
      #creating density estimation
        kde = KernelDensity(kernel = kernel, algorithm = 'ball_tree', bandwidth= 'silverman').fit(data)
        #print('dat:',data)
        examples = kde.sample(n_istances, random_state=0)
        
        

        print('KDE; Class: ', classe, 'Selected minority:',data.shape,'istances generated:', len(examples))
        print('New examples: ',examples.shape)
        print('Sampled:',examples)

      # INSERT THE ROWS HERE, DO NOT USE FUNCTION 
        list_samples.append((classe, examples)) 
        
    X,Y = augmenting_df(list_samples, X,Y)

    return X,Y
#this function oversamples raw data by SMOTE
#returns X data and Y labels with new added examples
def oversamp_SMOTE_definitive(X,Y):
  #matches number of istances in the maj class
  sm = SMOTE(random_state = 42, sampling_strategy = 'not majority') # resampling all mathcing the maj class (excluded)
  # generates new istances
  X_res , y_res = sm.fit_resample(X,Y)
  print('Oversampled with SMOTE:', Counter(y_res))
  
  return X_res,y_res

def cross_validation(X,Y,n_splits,model,kernel):
        
    #Provides the data in X,Y as np.array
    skf = StratifiedKFold(n_splits= n_splits, random_state=8, shuffle=True)
    skf.get_n_splits(X,Y)  #X is (n_samples; n_features) y (target variable)

    list_predict_prob_KDE, list_predict_prob_origi, list_predict_SMOTE = [], [], []   # initilize empty lists for probabilities
    list_acc_ori, list_acc_smote, list_acc_kde = [], [], []
    lst_y_test_labels = []
    lst_X_test = []

    for j,(train_index, test_index) in enumerate(skf.split(X,Y)):     #iterate over the number of splits, returns indexes of the train and test set
        
            x_train_fold, x_test_fold = X[train_index], X[test_index]      #selected data for train and test in j-fold
            y_train_fold, y_test_fold = Y[train_index], Y[test_index]      #selected targets
            
            print('\n')
            print('ITERATION --> ',j)
            print('Raw training data: ',x_train_fold.shape)
            print('Raw training labels: ',y_train_fold.shape)
            print('Testing data: ',x_test_fold.shape)
        #-----------------------------------------------------------------------
        # 1) STEP
        #call the method OversampKDE on the training partitioning, which
            #- given a list of class examples, oversamples the minority classes
            x1_fold,y1_fold= oversamp_KDE_definitive(x_train_fold,y_train_fold,kernel)
        #call the method OversampSMOTE on the training partitioning
            x2_fold,y2_fold = oversamp_SMOTE_definitive(x_train_fold,y_train_fold)

        #------------------------------------------------------------------------
        # 2) STEP
        #fit model on augmented dataset KDE
            model.fit(x1_fold,y1_fold)
            # --> test model on test fold; append predict_proba
            y_proba_kde = model.predict_proba(x_test_fold)
            y_pred_kde = model.predict(x_test_fold)
            acc_kde = accuracy_score(y_test_fold,y_pred_kde)

            list_acc_kde.append(acc_kde)
            list_predict_prob_KDE.append(y_proba_kde)
            #----------------------------
            #fit model on augmented training SMOTE
            model.fit(x2_fold,y2_fold)
            # test --< model on test fold ; append predict_proba
            y_proba_smote = model.predict_proba(x_test_fold)

            y_pred_smote = model.predict(x_test_fold)
            acc_smote = accuracy_score(y_test_fold,y_pred_smote)

            list_acc_smote.append(acc_smote)
            list_predict_SMOTE.append(y_proba_smote)
            #-----------------------------------
            #fit model on train set (normal)
            model.fit(x_train_fold,y_train_fold)

            # --> test model on test fold; append predict_proba
            y_proba = model.predict_proba(x_test_fold)

            y_pred= model.predict(x_test_fold)
            acc_ori= accuracy_score(y_test_fold,y_pred)

            list_acc_ori.append(acc_ori)
            list_predict_prob_origi.append(y_proba)
            #-------------------------------------
            # 3) STEP
            # append Y_test labels to lst_y_test_labels
            lst_y_test_labels.append(y_test_fold)

    #averaged accuracies over k-folds
    d_acc = {'model_original' : mean(list_acc_ori),'model_smote':mean(list_acc_smote),'model_kde':mean(list_acc_kde)}
    #aggregating k-fold probability estimates
    d_proba = {'model_original':np.concatenate(list_predict_prob_origi,axis = 0),'model_smote':np.concatenate(list_predict_SMOTE,axis = 0),'model_kde':np.concatenate(list_predict_prob_KDE,axis = 0)}  
    y_test_labels = np.concatenate(lst_y_test_labels, axis = 0)
    
    return y_test_labels, d_proba, d_acc

def result_render_multiclass(d_probabilities,d_accuracies,y_test):
  table_multi_micro = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])
  table_multi_macro = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])
  auc_micro, auc_macro, accuracy= [], [], []
  list_metrics = []

  classes = sorted(list(np.unique(y_test)))
  print('Sorted:',classes)
  n_classes = len(np.unique(y_test))

  y_test_binarize = label_binarize(y_test, classes=classes)
  print('Binarized:',y_test_binarize)
  #y_test_binarize = label_binarize(y_test, classes=np.arange(classes))

  scores = {}

  for model_name, model_proba in d_probabilities.items():  #iterating over 3 probabilities of 3 models
    y_pred = model_proba
    print('Predicted scores :',model_proba.shape)
    scores[model_name] = model_proba
    

    fpr ,tpr ,roc_auc ,thresholds = dict(), dict(), dict() ,dict() 
    # micro-average
    for i in range(n_classes):
      fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarize[:, i], y_pred[:, i], drop_intermediate=True)
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarize.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #aggregates all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    #mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    #print('All FPR:',all_fpr)
    tpr["macro"] = mean_tpr
    #print(mean_tpr)
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # storing average-micro fpr, tpr, auc for each method (original,smote,kde)
    row_micro = {'Classifier': model_name, 'fpr': fpr['micro'],'tpr':tpr['micro'],'auc':roc_auc['micro']}
    #row_micro = {'Classifier': model_name, 'fpr': fpr['micro'],'tpr':tpr['micro'],'auc':roc_auc['micro']}
    table_multi_micro.loc[len(table_multi_micro)] = row_micro

    # storing average-macro fpr, tpr, auc for each method (original,smote,kde)
    row_macro = {'Classifier': model_name,'fpr':fpr['macro'],'tpr':tpr['macro'],'auc':roc_auc['macro']}
    #row_macro = {'Classifier': model_name,'fpr':fpr['macro'],'tpr':tpr['macro'],'auc':roc_auc['macro']}
    table_multi_macro.loc[len(table_multi_macro)] = row_macro

    #appending AUC(ROC) for micro and macro average
    auc_micro.append(roc_auc_score(y_test, y_pred, multi_class='ovr',average = 'micro' ))
    auc_macro.append(roc_auc_score(y_test, y_pred, multi_class='ovr',average = 'macro' ))

  for acc in d_accuracies.values():  #appending average accuracies (over 10)  for raw,smote,kde to list:  3 accuracies
    accuracy.append(acc)
  for acc_score,auc_micro,auc_macro in zip(accuracy,auc_micro,auc_macro):  #creating list containing acc,auc,imcp for each method sequentially
    list_metrics.append(float(f'{acc_score:.3f}'))
    list_metrics.append(float(f'{auc_micro:.3f}'))
    list_metrics.append(float(f'{auc_macro:.3f}')) #auc micro  #inserted new auc !! macro

  return list_metrics, table_multi_macro, table_multi_micro

def multi_class_roc_save(title_set,table,model,save_folder,name = str()):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    #macro
    #plt.figure(dpi=600)
    plt.figure(figsize=(8,6))
    table.set_index('Classifier', inplace = True)
    #colors = ['navy','orange','green']
    colors = ['royalblue','saddlebrown','lightcoral']
    for i,color in zip(table.index,colors):
      plt.plot(table.loc[i]['fpr'], 
            table.loc[i]['tpr'], 
            label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']),color = color)
    plt.xlim([-0.005, 1.01])
    plt.ylim([-0.005, 1.01])
    plt.xticks([i/10.0 for i in range(11)])
    plt.yticks([i/10.0 for i in range(11)])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('{}-average ROC curve  - {}'.format(name, title_set), fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)

    file_name_macro = os.path.join(save_folder, '{}_{}_{}'.format(title_set,model,name))
    plt.savefig(file_name_macro)
    plt.close()


if __name__== '__main__':
    
    cv_splits = 10

    #classifier = GaussianNB()
    #classifier = DecisionTreeClassifier(random_state=0)
    classifier = RandomForestClassifier(random_state = 0)

    kernel = 'gaussian'
    binary = False  #if True, only binary; elif False multiclass (cumida multiclass and case-study)
    cumida = False #parameter for dataset pre processing
    value = True   #all variables are considered in the dataset
    
    source_folder = '../stack_exchange'
    save_folder_roc_curve= '../stack_exchange'

    all_files = glob.glob(source_folder +'/*.csv')
    
    for file in all_files:
        if file.endswith(".csv") :
            name = file.split('\\')[1]
            dataset = pd.read_csv(file)

            final_table_multiclass= pd.DataFrame(columns = ['Dataset','Accuracy (raw)','AUC (raw micro)','AUC (raw macro)','Accuracy (smote)','AUC (smote micro)','AUC (smote macro)','Accuracy(KDE)','AUC(KDE micro)','AUC (KDE macro)'])
            
            title = 'Iris_imbalanced'

            n_cols, classes = attr_count(dataset)
            print('--------------------- Statistics ------------------ DATASET --------------------------------')
            print('{}: \n Variables: {}'.format(title,n_cols))
            print('Classes:')
            print(classes)
            
            #returns data.array ad labels.array, attributes only on the basis of col selected, 
            x,y, variables = dataset_parser(dataset,value,cumida)
            print('Data: ',x.shape)
            print('Labels: ',y.shape)
            labels, probabilities_final, accuracies = cross_validation(x,y,cv_splits,classifier,kernel)  #added model choice

            list_metrics, table_multi_macro, table_multi_micro = result_render_multiclass(probabilities_final,accuracies,labels) #labels_name !!
            multi_class_roc_save(title,table_multi_macro,classifier,save_folder_roc_curve, name= 'Macro')
            multi_class_roc_save(title,table_multi_micro,classifier,save_folder_roc_curve,name = 'Micro')
            # The line below contains metrics such as: accuracy, auc and
            row_table = {'Dataset' : 'Iris' ,'Accuracy (raw)':list_metrics[0],'AUC (raw micro)':list_metrics[1],'AUC (raw macro)':list_metrics[2],'Accuracy (smote)':list_metrics[3],'AUC (smote micro)': list_metrics[4],'AUC (smote macro)':list_metrics[5],'Accuracy(KDE)':list_metrics[6],'AUC(KDE micro)':list_metrics[7],'AUC (KDE macro)':list_metrics[8]}    
            #final_table_multiclass.loc[len(final_table_multiclass)] = row_table
