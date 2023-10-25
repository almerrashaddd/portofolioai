import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# Load the Heart Disease dataset
dataset = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
heart_data = pd.read_csv(dataset)

# Handle missing values
heart_data = heart_data.dropna()

# After reduction
pca = PCA(n_components=9)
heart_data_reduced = pca.fit_transform(heart_data.drop('target', axis=1))

# Standardize the features
scaler = StandardScaler()
heart_data_scaled = scaler.fit_transform(heart_data_reduced)

# Load the Heart Disease dataset
X = heart_data_scaled
y = heart_data['target']

# Build the Decision Tree model with criterion='gini' and max_depth = 30
model = DecisionTreeClassifier(criterion='gini', max_depth=30)
model.fit(X, y)

# Function to predict the heart disease
def predict_heart_disease(features):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    if prediction == 0:
        pred = "No Heart Disease"
        return pred
    else:
        pred = "Yes Heart Disease"
        return pred

# Streamlit App
st.title("Heart Disease Prediction")
st.sidebar.header("Input Features")

# Define input elements for user input
ca = st.sidebar.slider("Number of Major Vessels (ca)", 0.0, 3.0, 3.0)
thal = st.sidebar.slider("Thalassemia Type (thal)", 0.0, 3.0, 2.0)
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80.0, 200.0, 80.0)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 3.0)
slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", 0.0, 2.0, 1.0)
restecg = st.sidebar.slider("Resting Electrocardiographic Results (restecg)", 0.0, 2.0, 1.0)
exang = st.sidebar.slider("Exercise-Induced Angina (exang)", 0.0, 1.0, 1.0)
chol = st.sidebar.slider("Serum Cholesterol (chol)", 80.0, 600.0, 80.0)
fbs = st.sidebar.slider("Fasting Blood Sugar (fbs)", 0.0, 1.0, 1.0)

input_features = [[ca, thal, trestbps, oldpeak, slope, restecg, exang, chol, fbs]]

# Make prediction
if st.sidebar.button("Predict"):
    prediction = predict_heart_disease(input_features)
    st.write("Predicted Class:", prediction)

st.write("Note: Adjust the input values in the sidebar and click 'Predict' to make a prediction.")
