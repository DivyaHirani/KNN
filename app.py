import streamlit as st
import joblib
import numpy as np

# Load
model = joblib.load("knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")

st.set_page_config(page_title="Employee Classifier")

st.title("Employee Performance Classification")

reports = st.slider("Reports Submitted",0,15,5)
closure = st.slider("Closure Rate %",0,100,60)
attendance = st.slider("Attendance %",0,100,85)
incentive = st.slider("Incentive Score",0,15000,3000)

if st.button("Predict Level"):

    data = np.array([[reports,closure,attendance,incentive]])
    data = scaler.transform(data)

    level = model.predict(data)[0]

    st.success(f"Predicted Performance Level: {level}")

    # Recommendation logic
    if level == 1:
        st.warning("Recommendation: Intensive training required")
    elif level == 2:
        st.info("Recommendation: Skill improvement program")
    elif level == 3:
        st.success("Recommendation: Eligible for incentives")
    else:
        st.balloons()
        st.success("Recommendation: Promotion potential")
