import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,auc

st.set_page_config(page_title='Diabetes Prediction',layout='wide',initial_sidebar_state='expanded')

# Load the pickle file
default_model = pickle.load(open(r'G:\Projects\Sugar_Prediction\pickle_files\model.pkl', 'rb'))
default_scaler = pickle.load(open(r'G:\Projects\Sugar_Prediction\pickle_files\scaler.pkl', 'rb'))
balanced_model = pickle.load(open(r'G:\Projects\Sugar_Prediction\pickle_files\rfc_model_smote.pkl', 'rb'))
balanced_scaler = pickle.load(open(r'G:\Projects\Sugar_Prediction\pickle_files\sc_1_smote.pkl', 'rb'))
tuned_model = pickle.load(open(r'G:\Projects\Sugar_Prediction\pickle_files\rfc_model_tuning.pkl', 'rb'))


# USer input from client
def user_input_features():
    pregnancies = st.sidebar.slider('Age',0,100,50)
    glucose = st.sidebar.slider('Glucose',0,300,50)
    blood_pressure = st.sidebar.slider('Blood Pressure',0,200,50)
    skin_thickness = st.sidebar.slider('Skin Thickness',0,300,50)
    insulin = st.sidebar.slider('Insulin',0,300,50)
    bmi = st.sidebar.slider('BMI',0,30,50)
    dpf = st.sidebar.slider('Diabetes Prediction Function',0.0,3.0,3.5)
    age = st.sidebar.slider('Age',0,100,20)

    data ={
        'Pregnancies':pregnancies,
        'Glucose': glucose,
        'BloodPressure':blood_pressure,
        'SkinThickness' :skin_thickness,
        'Insulin':insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction':dpf,
        'Age':age,
    }
    features = pd.DataFrame(data, index=[0])

    return features

input_df = user_input_features()

# Display The user input
st.subheader('User Input parameters')
st.write(input_df)

#Preprocessing the user input
input_df_scaled = default_scaler.transform(input_df)

# make Prediction
prediction =default_model.predict(input_df_scaled)[0]

# Display the prediction
if prediction ==0:
    st.subheader('Diabetic')
else:
    st.subheader('Non-Diabetic')


# Display the Prediction
st.subheader('Prediction',)





