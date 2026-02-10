import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee AI Dashboard",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")

# ---------------- CSS STYLE ----------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}

/* Card Style */
.card {
    background: rgba(0,0,0,0.55);
    padding:25px;
    border-radius:18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.6);
}

/* Title */
.title {
    font-size:42px;
    font-weight:800;
    text-align:center;
    color:#00ffe7;
}

/* Labels readable */
label, span, div {
    color:white !important;
}

/* Button Style */
button[kind="primary"] {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">📊 Employee Performance AI</div>', unsafe_allow_html=True)
st.write("KNN-powered classification dashboard")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,2])

# -------- INPUT PANEL --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Employee Inputs")

    reports = st.slider("Reports Submitted",0,15,5)
    closure = st.slider("Closure Rate %",0,100,60)
    attendance = st.slider("Attendance %",0,100,85)
    incentive = st.slider("Incentive Score",0,15000,3000)

    predict = st.button("Analyze Performance")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- OUTPUT PANEL --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI Evaluation")

    if predict:

        data = np.array([[reports,closure,attendance,incentive]])
        data = scaler.transform(data)
        level = model.predict(data)[0]

        confidence = min(100, (reports+closure+attendance)/3)

        st.progress(int(confidence))

        colA,colB,colC,colD = st.columns(4)
        colA.metric("Reports",reports)
        colB.metric("Closure",closure)
        colC.metric("Attendance",attendance)
        colD.metric("Incentive",incentive)

        if level == 1:
            st.error("🔴 Level 1 — Needs Improvement")
            st.warning("Recommendation: Intensive training")
        elif level == 2:
            st.warning("🟠 Level 2 — Developing")
            st.info("Recommendation: Skill improvement")
        elif level == 3:
            st.success("🟢 Level 3 — Strong Performer")
            st.success("Recommendation: Incentive eligible")
        else:
            st.balloons()
            st.success("🏆 Level 4 — Top Performer")
            st.success("Recommendation: Promotion candidate")

    else:
        st.info("Enter data and click Analyze")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Interactive HR Intelligence Dashboard")
