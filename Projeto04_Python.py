import pandas as pd
import numpy as np


#preprocessing packages
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder

#pipeline
from sklearn.pipeline import Pipeline

#model
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

#train/test split
from sklearn.model_selection import train_test_split

#model metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#resampling with  smote
from imblearn.over_sampling import SMOTE


#1) loading data
raw_dataframe_train = pd.read_csv('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\BigDataRealTimeAnalyticscomPythoneSpark\\Projeto4\\projeto4_telecom_treino.csv')
raw_dataframe_train['source']='train'

raw_dataframe_test = pd.read_csv('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\BigDataRealTimeAnalyticscomPythoneSpark\\Projeto4\\projeto4_telecom_teste.csv')
raw_dataframe_test['source']='test'

#checking columns names
np.array([raw_dataframe_train.columns!=raw_dataframe_test.columns]).sum()

#concatenating train and test

raw_dataframe = pd.concat([raw_dataframe_train,raw_dataframe_test])


#variable types
raw_dataframe.dtypes

predictive_variables = ['source','international_plan','voice_mail_plan','number_vmail_messages',
  'total_day_minutes','total_day_charge','total_eve_minutes','total_eve_charge',
  'total_night_minutes','total_night_charge','total_intl_minutes','total_intl_calls',
  'total_intl_charge','number_customer_service_calls'] 
  
#adding target variable  
predictive_variables = predictive_variables + ['churn']
  
  
cleaned_dataframe = raw_dataframe.copy()

cleaned_dataframe = cleaned_dataframe.loc[:, predictive_variables]


#obs: all axplanations for the feature selection were done in R EDA Script 





#1.1) Transforming number_vmail_messages variable
cleaned_dataframe['number_vmail_messages'] = np.where(cleaned_dataframe.number_customer_service_calls==0,'equal_zero','more_than_zero')

#1.2) Transfoming number_customer_service_calls 
cleaned_dataframe['number_customer_service_calls'] = np.where(cleaned_dataframe.number_customer_service_calls<4,'less_4','equal_more_than_4')

#1.3) Excluding total_intl_minutes 
cleaned_dataframe = cleaned_dataframe.drop('total_intl_minutes',axis = 1)

#1.4) Agreggating similar variables
cleaned_dataframe['total_minutes'] = cleaned_dataframe['total_day_minutes'] + cleaned_dataframe['total_eve_minutes'] + cleaned_dataframe['total_night_minutes']
cleaned_dataframe = cleaned_dataframe.drop(['total_day_minutes','total_eve_minutes','total_night_minutes'],axis = 1)

cleaned_dataframe['total_charge'] = cleaned_dataframe['total_day_charge'] + cleaned_dataframe['total_eve_charge'] + cleaned_dataframe['total_night_charge']
cleaned_dataframe = cleaned_dataframe.drop(['total_day_charge','total_eve_charge','total_night_charge'],axis = 1)

#1.5) Excluding total_charge
cleaned_dataframe = cleaned_dataframe.drop('total_charge', axis = 1)

#1.6) Excluding number_vmail_messages
cleaned_dataframe = cleaned_dataframe.drop('number_vmail_messages', axis = 1)

#1.7) Excluding total_intl_calls
cleaned_dataframe = cleaned_dataframe.drop('total_intl_calls', axis = 1)

#1.8) Excluding total_intl_charge
cleaned_dataframe = cleaned_dataframe.drop('total_intl_charge', axis = 1)

#1.9) TSetting 1 as our positive label (there was churn)

cleaned_dataframe['churn'] = np.where(cleaned_dataframe.churn=='no',0,1)




# 2) DATA RESAMPLING ----
#training set
cleaned_dataframe_training = cleaned_dataframe[cleaned_dataframe.source=='train']
cleaned_dataframe_training = cleaned_dataframe_training.drop('source', axis = 1)
X_train = cleaned_dataframe_training.copy()
X_train = X_train.drop('churn', axis = 1)
#X_train = X_train.to_numpy()
y_train = cleaned_dataframe_training[['churn']]


#testing set
cleaned_dataframe_testing= cleaned_dataframe[cleaned_dataframe.source=='test']
cleaned_dataframe_testing = cleaned_dataframe_testing.drop('source', axis = 1)
X_test = cleaned_dataframe_testing.copy()
X_test = X_test.drop('churn', axis = 1)
#X_test = X_test.to_numpy()
y_test = cleaned_dataframe_testing[['churn']]


#3) MODEL SPECIFICATION ----

#instantiate model
baseline_model = LogisticRegression()


#4) FEATURE ENGINEERING ----

numeric_recipe = Pipeline([('step_normalize', StandardScaler() )])

categorical_recipe = Pipeline([('step_dummy', OneHotEncoder() )])

categorical_mask = X_train.dtypes == 'object'

categorical_variables = X_train.columns[categorical_mask]
numeric_variables = X_train.columns[~categorical_mask]

recipe_baseline = ColumnTransformer(
    transformers=[
        ("num", numeric_recipe, numeric_variables),
        ("cat", categorical_recipe, categorical_variables),
    ]
)


#5) RECIPE TRAINING ----
recipe_prep_baseline = Pipeline(
    steps=[("recipe_baseline", recipe_baseline)]
)


recipe_prep_baseline.fit(X_train)

#6) PREPROCESS TRAINING DATA ----

X_train = recipe_prep_baseline.transform(X_train)


#resampling training data
smote_train = SMOTE()

# Create the resampled feature set
X_train, y_train = smote_train.fit_resample(X_train, y_train)


#7) PREPROCESS TEST DATA ----

X_test = recipe_prep_baseline.transform(X_test)


#8) MODELS FITTING ----
y_train = y_train.values.ravel()
baseline_model.fit(X_train,y_train) 

#9) PREDICTIONS ON TEST DATA ----
predictions_baseline = baseline_model.predict(X_test)

#confusion matrix
cm = confusion_matrix(y_test, predictions_baseline)
print(cm)

#precision
precision_testing = precision_score(y_test, predictions_baseline)
print("The precision value is {0:.2f}".format(precision_testing))

#recall
recall_testing = recall_score(y_test, predictions_baseline)
print("The recall value is {0:.2f}".format(recall_testing))


# Evaluate test-set roc_auc_score
predicion_proba_positive_class = baseline_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, predicion_proba_positive_class) 

# Print roc_auc_score
print('ROC AUC score: {:.3f}'.format(roc_auc))

