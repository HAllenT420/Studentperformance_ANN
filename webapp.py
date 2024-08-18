import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('student_model.keras')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Student Class Grade Prediction')


# User input
Age = st.slider('Age', 15, 18, value=15)  # Age from 15 to 18 years

Gender = st.selectbox('Gender', ['Male', 'Female'])  # Gender with descriptive labels
Gender = 1 if Gender == 'Female' else 0  # Convert to numeric representation

Ethnicity = st.selectbox('Ethnicity', 
                         ['Caucasian', 'African American', 'Asian', 'Other'])  # Ethnicity with descriptive labels
Ethnicity = ['Caucasian', 'African American', 'Asian', 'Other'].index(Ethnicity)  # Convert to numeric representation

ParentalEducation = st.selectbox('Parental Education',
                                 ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'])  # Education level with descriptive labels
ParentalEducation = ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'].index(ParentalEducation)  # Convert to numeric representation

StudyTimeWeekly = st.number_input('Weekly Study Time (hours)', 0, 20)  # Weekly study time from 0 to 20 hours

Absences = st.number_input('Number of Absences', 0, 30)  # Number of absences from 0 to 30

Tutoring = st.selectbox('Tutoring', ['No', 'Yes'])  # Tutoring status with descriptive labels
Tutoring = 1 if Tutoring == 'Yes' else 0  # Convert to numeric representation

ParentalSupport = st.selectbox('Parental Support',
                               ['None', 'Low', 'Moderate', 'High', 'Very High'])  # Parental support with descriptive labels
ParentalSupport = ['None', 'Low', 'Moderate', 'High', 'Very High'].index(ParentalSupport)  # Convert to numeric representation

Extracurricular = st.selectbox('Extracurricular Activities', ['No', 'Yes'])  # Extracurricular activities with descriptive labels
Extracurricular = 1 if Extracurricular == 'Yes' else 0  # Convert to numeric representation

Sports = st.selectbox('Sports', ['No', 'Yes'])  # Sports participation with descriptive labels
Sports = 1 if Sports == 'Yes' else 0  # Convert to numeric representation

Music = st.selectbox('Music', ['No', 'Yes'])  # Music participation with descriptive labels
Music = 1 if Music == 'Yes' else 0  # Convert to numeric representation

Volunteering = st.selectbox('Volunteering', ['No', 'Yes'])  # Volunteering with descriptive labels
Volunteering = 1 if Volunteering == 'Yes' else 0  # Convert to numeric representation

# Prepare the input data for scaling
input_data = np.array([[Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, 
                        Tutoring, ParentalSupport, Extracurricular, Sports, Music, 
                        Volunteering]])

# Scale the input data using the loaded scaler
scaled_data = scaler.transform(input_data)

# Prediction button
if st.button('Submit'):
    # Make a prediction using the trained model
    prediction = model.predict(scaled_data)
    
    # Map prediction to GradeClass
    grade_class_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_grade = grade_class_map[predicted_class]

    # Display the prediction
    st.write('### Prediction:')
    st.write(f'The predicted GradeClass is: {predicted_grade} ({predicted_class})')


# Display GradeClass mapping
st.write('### GradeClass Mapping:')
st.write('0: A (GPA >= 3.5)')
st.write('1: B (3.0 <= GPA < 3.5)')
st.write('2: C (2.5 <= GPA < 3.0)')
st.write('3: D (2.0 <= GPA < 2.5)')
st.write('4: F (GPA < 2.0)')
