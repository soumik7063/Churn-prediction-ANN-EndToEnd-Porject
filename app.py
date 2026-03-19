import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st

# load model and pkl files
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("Churn prediction application")
# user input

geography = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,100)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance',min_value=0.0)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})
input_data['HasCrCard'] = input_data['HasCrCard'].map({'Yes': 1, 'No': 0})
input_data['IsActiveMember'] = input_data['IsActiveMember'].map({'Yes': 1, 'No': 0})
# data preprocessing
geography_encoded = one_hot_encoder.transform(input_data[['Geography']]).toarray()

geo_df = pd.DataFrame(
    geography_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Combine
input_df = pd.concat([input_data.drop(columns=['Geography']), geo_df], axis=1)

# Ensure correct column order
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale
scaled_input = scaler.transform(input_df)

# prediction
if st.button('Predict'):
    prediction = model.predict(scaled_input)
    churn_probability = prediction[0][0]
    st.write(f'Churn Probability: {churn_probability:.2f}')

