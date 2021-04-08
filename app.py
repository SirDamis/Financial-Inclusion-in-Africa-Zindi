
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

st.write("""
# Financial Inclusion Prediction Application

This is a machine learning web application that predict who is mostly likely to have a bank account

Data is obtained from Zindi
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file ", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        country = st.sidebar.selectbox('Country', ('Rwanda', 'Tanzania', 'Kenya', 'Uganda'))
        year = st.sidebar.selectbox('Year', (2016, 2017, 2018))
        location_type = st.sidebar.selectbox('Location Type', ('Rural', 'Urban'))
        cellphone_access = st.sidebar.selectbox('Cellphone Access', ('Yes', 'No'))
        household_size = st.sidebar.slider('Household Size', 1, 50, 1)
        age_of_respondent = st.sidebar.slider('Age of Respondent', 1, 100, 16)
        gender_of_respondent = st.sidebar.selectbox('Gender of Respondent', ('Female', 'Male'))
        relationship_with_head = st.sidebar.selectbox('Relationship with Head', ('Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'))
        marital_status = st.sidebar.selectbox('Marital Status', ('Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated', 'Dont know'))
        education_level = st.sidebar.selectbox('Education Level', ('Primary education', 'No formal education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training', 'Other/Dont know/RTA'))
        data = {
            'country': country,
            'year': year,
            'location_type': location_type,
            'cellphone_access': cellphone_access,
            'household_size': household_size,
            'age_of_respondent': age_of_respondent,
            'gender_of_respondent': gender_of_respondent,
            'relationship_with_head': relationship_with_head,
            'marital_status': marital_status,
            'education_level': education_level
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combine user input features with the entire dataset
train_df = pd.read_csv('train_v2.csv')
train = train_df.drop(columns=['uniqueid', 'bank_account'])
df = pd.concat([input_df, train])

df_used = df.copy()[:1]
# Log transformation of skewed feature
df.age_of_respondent = np.log1p(df.age_of_respondent)

# One Hot Encoding
df = pd.get_dummies(df, columns=['country', 'relationship_with_head', 'marital_status', 'education_level'])

# Label Encoding
le = LabelEncoder()
cols = ['location_type', 'cellphone_access', 'gender_of_respondent', 'job_type']
for col in cols:
    df[col] =  le.fit_transform(df[col])
df = df[:1] #Select the  first row i.e input data row

#Display user input
st.subheader('User Input Features')
if uploaded_file is not None:
    st.write(df_used)
else:
    st.write('Using the sample data shown below')
    st.write(df_used)
    
#Read saved classification model
pickle_in = open("inclusion__classifier.pkl", 'rb')
load_clf = pickle.load(pickle_in)

#Apply model to make predictions
prediction = load_clf.predict_proba(df)[:, 1]
st.subheader('Prediction Result')
if prediction >= 0.5:
    st.write('The user has a bank account.', 1)
else:
    st.write('The user does not have a bank account.', 0)
