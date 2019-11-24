#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from factor_analyzer import factor_analyzer, FactorAnalyzer
from sklearn.decomposition import PCA
import import_ipynb
import SplitData as SD
import lstm as lstm
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pdfkit as pdf
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import pickle
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset
from sklearn.model_selection import train_test_split


# # Load Data

# In[ ]:
inputs = {}
for file_name in os.listdir('./New_Data/'):
    if file_name.startswith('.'):
        continue
    name = file_name.replace('.csv', '')
    path = './New_Data/' + file_name
    inputs[name] = pd.read_csv(path, index_col = 0)


# ## Processing Data

# In[ ]:


SD.splitActivities(inputs)
SD.splitAudio(inputs)
SD.splitDark(inputs)
SD.splitConversation(inputs)


# ### Load Processed Data

# In[ ]:


inputs = {}
for file_name in os.listdir('./Final_Data/'):
    if file_name.startswith('.'):
        continue
    name = file_name.replace('.csv', '')
    path = './Final_Data/' + file_name
    df = pd.read_csv(path, index_col = 0)
    inputs[name]= df.replace(0, np.nan)
inputs['sms'] = inputs.pop('sms_spark')
inputs['call_log'] = inputs.pop('call_log_spark2')


# In[ ]:


def fillMissing(df):
    df = df.transpose().interpolate(method='linear').transpose()
    return df.transpose().interpolate(method = 'linear', limit_direction='backward').transpose()


# In[ ]:


for key in inputs.keys():
    if inputs[key].shape[1] != 1:
        inputs[key] = fillMissing(inputs[key])


# ### Split into positive negative and flourishing score dataframes

# In[ ]:


flourishing = pd.read_csv('./StudentLife_Dataset/Outputs/FlourishingScale.csv')
panas = pd.read_csv('./StudentLife_Dataset/Outputs/panas.csv')
positive_score=['uid', 'Interested', 'Strong', 'Enthusiastic', 'Proud', 'Alert', 'Inspired', 'Determined ', 'Attentive', 'Active ']
negative_score=['uid', 'Distressed', 'Upset', 'Guilty', 'Scared', 'Hostile ', 'Irritable','Nervous', 'Jittery', 'Afraid ']
df_flour_post = pd.DataFrame()
df_pos_post = pd.DataFrame()
df_neg_post = pd.DataFrame()
   
for i in range(60):
    temp_flour_post = (flourishing.loc[flourishing['uid'] == 'u' + str(f"{i:02d}")].loc[flourishing['type'] == 'post']).drop(columns='type')
    df_flour_post = pd.concat([df_flour_post, temp_flour_post], axis = 0)
    
    temp_post = panas.loc[panas['uid'] == 'u' + str(f"{i:02d}")].loc[panas['type'] == 'post']
    df1_post = temp_post[positive_score]
    df2_post = temp_post[negative_score]
    df_pos_post = pd.concat([df_pos_post, df1_post], axis=0)
    df_neg_post = pd.concat([df_neg_post, df2_post], axis=0)
df_flour_post = df_flour_post.set_index(keys='uid')
df_pos_post = df_pos_post.set_index(keys='uid')
df_neg_post = df_neg_post.set_index(keys = 'uid')
df_flour_post = df_flour_post.dropna()
df_pos_post = df_pos_post.dropna()
df_neg_post = df_neg_post.dropna()


# ### INPUT DATA for Flourishing Data###

# In[ ]:


## Look for people quit for post flourishing score testing ##
full_ids = []
for i in range(60):
    full_ids.append('u' + str(f"{i:02d}"))
ids_flour_post = df_flour_post.index.to_numpy()
quit_ids = list(set(full_ids) - set(ids_flour_post))
#print(quit_ids)
### Delete people quit, from dataframe ###
input_keys = inputs.keys()
flour_input = {}
for key in input_keys:
    flour_input[key] = inputs[key].drop(quit_ids, errors='ignore')


# # METHODS #

# ### BINARIZATION ###

# In[ ]:


### Convert score to binary data ###
def binarize(df, threshold):
    m = threshold
    if m < 1:
        df[df.iloc[:, 0] > m] = 1
        df[df.iloc[:, 0] <= m] = 0
    else:
        df[df.iloc[:, 0] <= m] = 0
        df[df.iloc[:, 0] > m] = 1
    return df


# ## Method 1
# In[ ]:

# ### LSTM 
lstm.runLSTM('flour')
lstm.runLSTM('pos')
lstm.runLSTM('neg')

# ## Method 2

# ### Input and Output
# In[ ]:


## Look for people quit for post flourishing score testing ##
full_ids = []
for i in range(60):
    full_ids.append('u' + str(f"{i:02d}"))
ids_flour_post = df_flour_post.index.to_numpy()
quit_ids = list(set(full_ids) - set(ids_flour_post))

### Delete people quit, from dataframe ###
input_keys = inputs.keys()
flour_input = {}
for key in input_keys:
    flour_input[key] = inputs[key].drop(quit_ids, errors='ignore')
for key in flour_input.keys():
    flour_input[key] = flour_input[key].sum(axis=1).to_frame()
X_flour = pd.DataFrame(columns=None)

## input and label for flourishing score
for key in flour_input:
    flour_input[key].columns = [key]
    X_flour = pd.concat([X_flour, flour_input[key]], axis = 1)
label_flour = binarize(df_flour_post.sum(axis = 1).to_frame(0), 44)


# In[ ]:


## Look for people quit for post flourishing score testing ##
full_ids = []
for i in range(60):
    full_ids.append('u' + str(f"{i:02d}"))
ids_pos_post = df_pos_post.index.to_numpy()
quit_ids = list(set(full_ids) - set(ids_pos_post))

### Delete people quit, from dataframe ###
input_keys = inputs.keys()
pos_input = {}
for key in input_keys:
    pos_input[key] = inputs[key].drop(quit_ids, errors='ignore')
for key in pos_input.keys():
    pos_input[key] = pos_input[key].sum(axis=1).to_frame()
X_pos = pd.DataFrame(columns=None)

## input and label for flourishing score
for key in pos_input:
    pos_input[key].columns = [key]
    X_pos = pd.concat([X_pos, pos_input[key]], axis = 1)
label_pos = binarize(df_pos_post.sum(axis = 1).to_frame(0), 29)


# In[ ]:


## Look for people quit for post flourishing score testing ##
full_ids = []
for i in range(60):
    full_ids.append('u' + str(f"{i:02d}"))
ids_neg_post = df_neg_post.index.to_numpy()
quit_ids = list(set(full_ids) - set(ids_neg_post))

### Delete people quit, from dataframe ###
input_keys = inputs.keys()
neg_input = {}
for key in input_keys:
    neg_input[key] = inputs[key].drop(quit_ids, errors='ignore')
for key in pos_input.keys():
    neg_input[key] = neg_input[key].sum(axis=1).to_frame()
X_neg = pd.DataFrame(columns=None)

## input and label for flourishing score
for key in neg_input:
    neg_input[key].columns = [key]
    X_neg = pd.concat([X_neg, neg_input[key]], axis = 1)
label_neg = binarize(df_neg_post.sum(axis = 1).to_frame(0), 16)


# ### KNN

# In[ ]:


def KNN(features, y, n_neighbour):
    
    X_train, X_test, y_train, y_test = train_test_split(features, y, train_size=0.8)
    neighbour = KNeighborsClassifier(n_neighbors=n_neighbour).fit(X_train, y_train)
    pred = neighbour.predict(X_test)
    roc_auc = 0
    acc = 0
    precision = 0
    recall = 0
    fscore = 0
    try:
        roc_auc = roc_auc_score(y_test, pred)
        acc = accuracy_score(y_test, pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, pred, average='weighted')
    except ValueError:
        _, _, _, _, roc_auc = KNN(features, y, n_neighbour)
        acc, _, _, _, _ = KNN(features, y, n_neighbour)
        _, precision, recall, fscore, _ =KNN(features, y, n_neighbour)
      
    return [acc, precision, recall, fscore, roc_auc]


# ### Optimisation

# In[ ]:


### ROC and AUC to find the optimal k ###
def roc_auc_comparison(features, y):
    n = 4
    kf = KFold(n_splits=n, shuffle=True)
    scores = []
    for i in range(2,10):
        score = 0
        n = 0
        for train, test in kf.split(features):
            X_train, X_test, y_train, y_test = features[train, :], features[test, :], y[train], y[test]
            model = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:,1]
            try:
                score += roc_auc_score(y_test, pred)
                n += 1
            except ValueError:
                pass
            
        scores.append(score/n)
    n_neighbour = np.asarray(scores).argmax()+2
    plt.plot(range(2,10),scores,label='AUC_score',color='grey')
    plt.xlabel('number of neighbours')
    plt.ylabel('AUC score')
    return n_neighbour


# ### Feature Selection

# In[ ]:


def KNN_feature_selection(X, label, name):
    count = [0] *len(X.columns)
    for i in range(100):
        model = RandomForestClassifier()
        rfe = RFE(model)
        rfe = rfe.fit(X, label)
        a = rfe.support_ 

        X_new = rfe.transform(X)
        X_new = pd.DataFrame(X_new, index = X.index, columns=X.columns[a])
        n = roc_auc_comparison(features=X_new.to_numpy(), y=label.to_numpy())
        acc, _, _, _, _ = KNN(X_new, y=label, n_neighbour=n)
        if acc <= 0.5:
            continue
        for j in range(len(rfe.ranking_)):
            if rfe.ranking_[j] == 1:
                count[j] += 1
    plt.show()
    temp = pd.DataFrame(count, index = X.columns, columns=['freq']).sort_values(by = ['freq'], ascending=False)
    plt.barh(temp.index, temp.iloc[:,0])
    title = 'Feature Importance for ' + name + ' score in KNN'
    plt.title(title)
    path = './Images/knn_importance_' + name + '.png'
    plt.xlabel('Frequency')
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    return np.array(count).argsort()[-5:]


# #### Flourishing score

# In[ ]:


chosen = KNN_feature_selection(X_flour, label_flour, 'flour')
X_flour_chosen = X_flour.iloc[:, chosen]
#X_flour_chosen


# #### Positive Score

# In[ ]:


chosen = KNN_feature_selection(X_pos, label_pos, 'pos')
X_pos_chosen = X_pos.iloc[:, chosen]


# In[ ]:


chosen = KNN_feature_selection(X_neg, label_neg, 'neg')
X_neg_chosen = X_neg.iloc[:, chosen]


# ### Evaluation

# #### Flourishing Score

# In[ ]:


score = pd.DataFrame(columns = None)
optimalK = []
for i in range(100):
    n = roc_auc_comparison(X_flour_chosen.to_numpy(), label_flour.to_numpy())
    optimalK.append(n)
    li = KNN(X_flour_chosen.to_numpy(), label_flour.to_numpy(), n)
    score = pd.concat([score, pd.DataFrame(np.array(li))], axis=1)
plt.title('Find Optimal K for KNN for predicting Flourishing Score')
plt.savefig('./Images/k_flour.png', bbox_inches='tight')
plt.show()

temp = np.unique(np.array(optimalK),return_counts=True)
index = temp[0]
freq = temp[1]
#index, freq
plt.barh(index , freq)
plt.title('Frequency of each K selected to be an optimal for Flourishing Score in KNN')
plt.ylabel('K')
plt.xlabel('frequency')
path = './Images/knn_optimalK_four.png'
plt.savefig(path, bbox_inches='tight')
plt.show()
score.index = ['acc', 'precision', 'recall', 'fscore', 'roc_auc']


# In[ ]:


temp = score.mean(axis = 1).to_frame()
temp.columns = ['Score for Flourishing']
temp


# #### Positive Score

# In[ ]:


score = pd.DataFrame(columns = None)
optimalK = []
for i in range(100):
    n = roc_auc_comparison(X_pos_chosen.to_numpy(), label_pos.to_numpy())
    optimalK.append(n)
    li = KNN(X_pos_chosen.to_numpy(), label_pos.to_numpy(), n)
    score = pd.concat([score, pd.DataFrame(np.array(li))], axis=1)
plt.title('Find Optimal K for KNN for predicting Positive Score')
plt.savefig('./Images/k_positive.png', bbox_inches='tight')
plt.show()

temp = np.unique(np.array(optimalK),return_counts=True)
index = temp[0]
freq = temp[1]
#index, freq
plt.barh(index , freq)
plt.title('Frequency of each K selected to be an optimal for Positive Score in KNN')
plt.ylabel('K')
plt.xlabel('frequency')
path = './Images/knn_optimalK_pos.png'
plt.savefig(path, bbox_inches='tight')
plt.show()
score.index = ['acc', 'precision', 'recall', 'fscore', 'roc_auc']


# In[ ]:


temp = score.mean(axis = 1).to_frame()
temp.columns = ['Score for PANAS Positive']
temp


# #### Negative Score

# In[ ]:


score = pd.DataFrame(columns = None)
optimalK = []
for i in range(100):
    n = roc_auc_comparison(X_neg_chosen.to_numpy(), label_neg.to_numpy())
    optimalK.append(n)
    li = KNN(X_neg_chosen.to_numpy(), label_neg.to_numpy(), n)
    score = pd.concat([score, pd.DataFrame(np.array(li))], axis=1)
    
    
plt.title('Find Optimal K for KNN for predicting Negative Score')
plt.savefig('./Images/k_negative.png', bbox_inches='tight')
plt.show()


temp = np.unique(np.array(optimalK),return_counts=True)
index = temp[0]
freq = temp[1]
#index, freq
plt.barh(index , freq)
plt.title('Frequency of each K selected to be an optimal for Negative Score in KNN')
plt.ylabel('K')
plt.xlabel('frequency')
path = './Images/knn_optimalK_neg.png'
plt.savefig(path, bbox_inches='tight')
plt.show()
score.index = ['acc', 'precision', 'recall', 'fscore', 'roc_auc']


# In[ ]:


temp = score.mean(axis = 1).to_frame()
temp.columns = ['Score for PANAS Negative']
temp


# ## Method 3

# ### Random Forest 

# ### Optimisation

# In[ ]:


def random_forest_validation(X, y, name, rang, score_type):
    train_score, validation_score = validation_curve(RandomForestClassifier(), X, y.to_numpy().ravel(), param_name=name,param_range=rang, scoring="accuracy", cv=3)
    sum_score = validation_score.sum(axis = 1)
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_score, axis=1)
    train_std = np.std(train_score, axis=1)

    # Calculate mean and standard deviation for test set scores
    validation_mean = np.mean(validation_score, axis=1)
    validation_std = np.std(validation_score, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(rang, train_mean, label="Training score", color="red")
    plt.plot(rang, validation_mean, label="Cross-validation score", color="blue")

    # Plot accurancy bands for training and test sets
    plt.fill_between(rang, train_mean - train_std, train_mean + train_std, color="pink")
    plt.fill_between(rang, validation_mean - validation_std, validation_mean + validation_std, color="lightblue")
    n = sum_score.argmax()
    print(sum_score[n])
    
    # Create plot
    title = "Validation Curve With Random Forest for " + score_type
    plt.title(title)
    plt.xlabel(name)
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    path = './Images/' + name + '_' + score_type + '.png'
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    return rang[n]


# In[ ]:


def optimiseRandomForest(X, label, score_type):
    rang = np.arange(1, 100, 2)
    name = "n_estimators"
    n_estimators = random_forest_validation(X, label, name, rang, score_type)
    rang = np.arange(2, 10, 1)
    name = "max_depth"
    max_depth = random_forest_validation(X, label, name, rang, score_type)
    rang = np.arange(2, 10, 1)
    name = "min_samples_split"
    min_samples_split = random_forest_validation(X, label,name, rang, score_type)
    rang = np.arange(2, 10, 1)
    name = "min_samples_leaf"
    min_samples_leaf = random_forest_validation(X, label,name, rang, score_type)
    return n_estimators, max_depth, min_samples_split, min_samples_leaf


# #### Flourishing Score

# In[ ]:


n_estimators, max_depth, min_samples_split, min_samples_leaf = optimiseRandomForest(X_flour, label_flour, 'flour')
X_flour_train, X_flour_valid, label_flour_train, label_flour_valid =  train_test_split(X_flour, label_flour, train_size=0.8)
print(n_estimators, max_depth, min_samples_split, min_samples_leaf)
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf)
clf.fit(X_flour_train, label_flour_train)
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index = X_flour.columns.to_numpy())
importance.columns = ['importance']
importance = importance.sort_values(by = ['importance'], ascending=False)
plt.barh(importance.index, importance.iloc[:, 0])
plt.title('Feature Importance for Flourishing Score')
plt.xlabel('importance rate')
plt.savefig('./Images/random_importance_flour.png', bbox_inches='tight')
plt.show()


# In[ ]:


pred = clf.predict(X_flour_valid)
acc = accuracy_score(label_flour_valid, pred)
precision, recall, fscore, _ = precision_recall_fscore_support(label_flour_valid, pred, average='weighted')
auc = roc_auc_score(label_flour_valid, pred)


# In[ ]:


pd.DataFrame(np.array([acc, precision, recall, fscore, auc]), index = ['accuracy', 'precision', 'recall', 'fscore', 'auc'], columns = ['Scores for Flourishing'])


# #### Positive Score

# In[ ]:


### Optimisation ###
X_pos_train, X_pos_valid, label_pos_train, label_pos_valid =  train_test_split(X_pos, label_pos, train_size=0.8)
n_estimators, max_depth, min_samples_split, min_samples_leaf = optimiseRandomForest(X_pos, label_pos, 'positive')

print(n_estimators, max_depth, min_samples_split, min_samples_leaf)
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf, random_state=0)
clf.fit(X_pos_train, label_pos_train)
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index = X_pos_train.columns.to_numpy())
importance.columns = ['importance']
importance = importance.sort_values(by = ['importance'], ascending=False)
plt.barh(importance.index, importance.iloc[:, 0])
plt.title('Feature Importance for Positive Score')
plt.xlabel('importance rate')
plt.savefig('./Images/random_importance_positive.png', bbox_inches='tight')
plt.show()


# In[ ]:


pred = clf.predict(X_pos_valid)
acc = accuracy_score(label_pos_valid, pred)
precision, recall, fscore, _ = precision_recall_fscore_support(label_pos_valid, pred, average='weighted')
auc = roc_auc_score(label_pos_valid, pred)


# In[ ]:


pd.DataFrame(np.array([acc, precision, recall, fscore, auc]), index = ['accuracy', 'precision', 'recall', 'fscore', 'auc'], columns = ['Scores for Positive'])


# #### Negative Scores

# In[ ]:


### Optimisation ###
X_neg_train, X_neg_valid, label_neg_train, label_neg_valid =  train_test_split(X_neg, label_neg, train_size=0.8)
n_estimators, max_depth, min_samples_split, min_samples_leaf = optimiseRandomForest(X_neg, label_neg, 'negative')

print(n_estimators, max_depth, min_samples_split, min_samples_leaf)
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf, random_state=0)
smt = SMOTE()
X_neg_train, label_neg_train = smt.fit_sample(X_neg_train, label_neg_train)

clf.fit(X_neg_train, label_neg_train)
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index = X_neg.columns.to_numpy())
importance.columns = ['importance']
importance = importance.sort_values(by = ['importance'], ascending=False)
plt.barh(importance.index, importance.iloc[:, 0])
plt.title('Feature Importance for Negative Score')
plt.savefig('./Images/random_importance_negative.png', bbox_inches='tight')
plt.show()


# In[ ]:


pred = clf.predict(X_neg_valid)
acc = accuracy_score(label_neg_valid, pred)
precision, recall, fscore, _ = precision_recall_fscore_support(label_neg_valid, pred, average='weighted')
auc = roc_auc_score(label_neg_valid, pred)


# In[ ]:


pd.DataFrame(np.array([acc, precision, recall, fscore, auc]), index = ['accuracy', 'precision', 'recall', 'fscore', 'auc'], columns = ['Scores for Negative'])


# ## Correlation 

# ### Flourishing Score

# In[ ]:


pd.concat([pd.DataFrame(X_flour['dark_time']), label_flour], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_flour['conversation_time']), label_flour], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_flour['call_log']), label_flour], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_flour['noise']), label_flour], axis = 1).corr()


# ### Negative Score

# In[ ]:


pd.concat([pd.DataFrame(X_neg['dark_freq']), label_neg], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_neg['dark_time']), label_neg], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_neg['conversation_time']), label_neg], axis = 1).corr()


# In[ ]:


pd.concat([pd.DataFrame(X_neg['walk']), label_neg], axis = 1).corr()


# ### positive score

# In[ ]:


for key in X_pos.keys():
    a = pd.concat([pd.DataFrame(X_pos[key]), label_pos], axis = 1).corr()
    path = './Images/' + key+'.html'
    a.to_html(path)







# %%
