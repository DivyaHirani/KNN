import streamlit as st
import joblib
import numpy as np

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Employee AI Dashboard", layout="wide")

# -------- LOAD MODEL --------
model = joblib.load("knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")

# -------- CUSTOM CSS --------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(120deg,#1e3c72,#2a5298);
    color:white;
}

/* Card Style */
.card {
    background: rgba(255,255,255,0.12);
    padding:25px;
    border-radius:20px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
}

/* Title */
.title {
    font-size:42px;
    font-weight:800;
    text-align:center;
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(90deg,#ff7e5f,#feb47b);
    border:none;
    border-radius:10px;
    font-size:18px;
}

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}

.card {
    background: rgba(0,0,0,0.55);
    padding:25px;
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)



# -------- OUTPUT CARD --------
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

        # Color-coded result
        if level == 1:
            st.error("🔴 Level 1 — Needs Improvement")
            st.warning("Recommendation: Intensive training required")
        elif level == 2:
            st.warning("🟠 Level 2 — Developing")
            st.info("Recommendation: Skill improvement program")
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

# -------- FOOTER --------
st.markdown("---")
st.caption("Interactive HR Intelligence Dashboard")


