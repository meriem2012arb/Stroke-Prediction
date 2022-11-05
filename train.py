#!/usr/bin/env python
# coding: utf-8



# ### Import Packages

import warnings
warnings.filterwarnings("ignore")


import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import roc_auc_score,mutual_info_score
from sklearn.linear_model import LogisticRegression




# Data Loading 
df = pd.read_csv("healthcare-dataset-stroke-data.csv")



# Exclude the 'id' Column (we noticed that it's unnecessary column) 
df= df.loc[:,df.columns != 'id']




df.bmi.fillna(np.mean(df.bmi), inplace = True)

categorical = ["gender", "work_type","Residence_type","smoking_status", "ever_married" ]
numerical = ["age","hypertension", "heart_disease", "avg_glucose_level", "bmi",
             "hypertension", "heart_disease"]


#
# ### Target feature : stroke



df['stroke'].value_counts()



#  ### Setting up the validation framework
# 
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.stroke.values
y_val = df_val.stroke.values
y_test = df_test.stroke.values



del df_train['stroke']
del df_val['stroke']
del df_test['stroke']


# ### One-hot encoding

# In[30]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)




# ## Model training

# ### Logistic regression 


def evaluation_model(model):
    
    # Training Dataset 
    y_pred = model.predict_proba(X_train)[:, 1]
    acc_train = roc_auc_score(y_train, y_pred)* 100
    print('Training Accuracy: ',acc_train)

    # Validation Dataset
    y_pred = model.predict_proba(X_val)[:, 1]
    acc_val = roc_auc_score(y_val, y_pred)* 100
    print('Validation Accuracy: ',acc_val )
    
    return {'acc_train' :acc_train,
            'acc_val' : acc_val }



# ### Training the final model 

# In[59]:


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.stroke.values
del df_full_train['stroke']

dict_final = df_full_train.to_dict(orient='records')

dv_final = DictVectorizer(sparse=False)

X_full_train = dv_final.fit_transform(dict_final)


final_model = LogisticRegression(C=0.05179474679231213, max_iter=200)
final_model.fit(X_full_train, y_full_train)



#evaluation_model(final_model)


# ## Save the modele
model_file = 'model00.bin'
dv_path = 'dv.bin'

with open(model_file, 'wb') as f_out:
    pickle.dump(final_model, f_out)

with open(dv_path, 'wb') as f_out:
    pickle.dump(dv_final, f_out)










