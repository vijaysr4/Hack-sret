#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import plotly.express as ex
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# In[3]:


# Load dataset
df = pd.read_csv("water_potability.csv")


# In[4]:


for col in ['Sulfate','ph','Trihalomethanes']:
    missing_label_0 = df.query('Potability == 0')[col][df[col].isna()].index
    df.loc[missing_label_0,col] = df.query('Potability == 0')[col][df[col].notna()].mean()

    missing_label_1 = df.query('Potability == 1')[col][df[col].isna()].index
    df.loc[missing_label_1,col] = df.query('Potability == 1')[col][df[col].notna()].mean()            
                                                                   
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[6]:


# Splititng Train and Test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)


# In[7]:


# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[8]:


# Light Gradient Boosting Machine Classifier
from lightgbm import LGBMClassifier
classifier = LGBMClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[11]:


from flask import Flask , render_template, url_for, redirect, request

app = Flask(__name__) 

@app.route("/")
def home():
    return render_template('Home.html')

@app.route("/dataform", methods=["POST", "GET"])
def dataform():
    return render_template('Dataform.html')

@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == 'POST':

        Dict = {"Ph" : request.form.get("water_Ph"),
                "Hardness" : request.form.get("water_Hardness"),
                "Solids" : request.form.get("water_Solids"),
                "Chloramine" : request.form.get("water_Chloramine") ,
                "Sulfate" : request.form.get("water_Sulfate"),
                "Conductivity" : request.form.get("water_Conductivity"),
                "Organic_Carbon" : request.form.get("water_Organic_Carbon"),
                "Trihalomethanes" : request.form.get("water_Trihalomethanes"),
                "water_Turbidity" : request.form.get("water_Turbidity"),
                }
 
        print("=================")
        print(Dict["Ph"])
        print(Dict["Hardness"])
        print(Dict["Solids"])
        print(Dict["Chloramine"])
        print(Dict["Sulfate"])
        print(Dict["Conductivity"])
        print(Dict["Organic_Carbon"])
        print(Dict["Trihalomethanes"])
        print(Dict["water_Turbidity"])
        print("=================")
        
        Result = classifier.predict(sc.transform(int(Dict["Ph"]), int(Dict["Hardness"]), int(Dict["Solids"]), int(Dict["Chloramine"]), int(Dict["Sulfate"]), int(Dict["Conductivity"]),int(Dict["Organic_Carbon"]), int(Dict["Trihalomethanes"]), int(Dict["water_Turbidity"]) ))

        return render_template('Result.html', pt = "Water Is Drinkable") if Result == 1 else render_template('result.html', pt = "Water is Not Drinkable")
 
app.run(debug = True)
 

