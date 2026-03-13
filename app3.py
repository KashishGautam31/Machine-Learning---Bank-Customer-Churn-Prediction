import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# CUSTOM CSS

page_bg = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"]{
font-family:'Poppins', sans-serif;
color:white;
}

/* BACKGROUND */

.stApp{
background-image:url("https://fintechweekly.s3.amazonaws.com/article/721/Revolut_Moves_to_Enter_Argentina_s_Banking_Sector_with_Planned_Acquisition_of_Banco_Cetelem.png");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

/* MAIN CONTAINER */

section.main > div{
background:rgba(0,0,0,0.35);
padding:30px;
border-radius:15px;
}

/* TITLE */

h1{
color:#00E5FF;
font-size:70px !important;
text-align:center;
font-weight:800;
}

/* TAB HEADINGS */

h2{
color:#4FC3F7;
font-size:48px !important;
font-weight:700;
}

/* SUB HEADINGS */

h3{
color:#81D4FA;
font-size:38px !important;
font-weight:700;
}

/* TEXT */

p,label{
color:white !important;
font-size:22px !important;
}

/* DATAFRAME CONTAINER */

[data-testid="stDataFrame"]{
background:white !important;
border-radius:10px;
padding:4px;
}

/* TABLE FONT SIZE */

[data-testid="stDataFrame"] *{
font-size:38px !important;
}

/* TABLE BORDER */

table{
border-collapse:collapse !important;
}

table, th, td{
border:0.5px solid #e0e0e0 !important;
}

/* INPUT BOXES */

.stNumberInput input{
background:rgba(255,255,255,0.12) !important;
color:white !important;
font-size:20px !important;
border-radius:8px;
}

/* SELECT BOX */

div[data-baseweb="select"] > div{
background:rgba(255,255,255,0.12) !important;
color:white !important;
font-size:20px !important;
}

/* SLIDER */

.stSlider span{
color:white !important;
font-size:18px;
}

/* BUTTON */

.stButton>button{
background:linear-gradient(90deg,#00c6ff,#0072ff);
color:white;
font-size:22px;
font-weight:bold;
height:60px;
width:240px;
border-radius:10px;
border:none;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# LOAD DATASET

df = pd.read_csv("Churn_Modelling.csv")

# LOAD MODEL

model = pickle.load(open("churn_model.pkl","rb"))

# TITLE

st.title("🏦 Bank Customer Churn Prediction")

st.write("Predict whether a bank customer will leave the bank.")

st.divider()

# TABS

tab1, tab2, tab3 = st.tabs([
"📊 Dataset Overview",
"📈 Data Visualization",
"🤖 Prediction"
])

# TAB 1

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

# TAB 2

with tab2:

    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots(figsize=(10,6))

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    sns.countplot(
    x="Exited",
    data=df,
    color="#00E5FF",
    edgecolor="black",
    linewidth=1.5,
    alpha=0.8,
    ax=ax
)

    ax.set_title("Customer Churn Count", color="white", fontsize=26)
    ax.tick_params(colors="white")

    ax.set_xlabel("Exited", color="white", fontsize=20)
    ax.set_ylabel("Number of Customers", color="white", fontsize=20)

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader("Age Distribution")

    fig2, ax2 = plt.subplots(figsize=(10,6))

    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")

    sns.histplot(df["Age"], kde=True, color="#00E5FF", ax=ax2)

    ax2.set_title("Age Distribution of Customers", color="white", fontsize=26)
    ax2.tick_params(colors="white")

    ax2.set_xlabel("Age", color="white", fontsize=20)
    ax2.set_ylabel("Frequency", color="white", fontsize=20)

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

    st.subheader("Balance Distribution")

    fig3, ax3 = plt.subplots(figsize=(10,6))

    fig3.patch.set_alpha(0)
    ax3.set_facecolor("none")

    sns.histplot(df["Balance"], kde=True, color="#FFA726", ax=ax3)

    ax3.set_title("Balance Distribution", color="white", fontsize=26)
    ax3.tick_params(colors="white")

    ax3.set_xlabel("Balance", color="white", fontsize=20)
    ax3.set_ylabel("Frequency", color="white", fontsize=20)

    ax3.grid(alpha=0.3)

    st.pyplot(fig3)

# TAB 3: PREDICTION
with tab3:
    st.subheader("Enter Customer Details")

    # Columns for input
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 750)           # example for "stay"
        geography = st.selectbox("Geography", ("France", "Spain", "Germany"), index=0)  # France = stay
        gender = st.selectbox("Gender", ("Male", "Female"), index=0)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure", 0, 10, 7)

    with col2:
        balance = st.number_input("Balance", 0.0, 250000.0, 100000.0)
        products = st.slider("Number of Products", 1, 4, 2)
        credit_card = st.selectbox("Has Credit Card", (0, 1), index=1)
        active_member = st.selectbox("Is Active Member", (0, 1), index=1)
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 80000.0)

    # Encoding
    gender_encoded = 1 if gender == "Male" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    # France is baseline (drop_first=True in training)

    # Feature array in exact same order as training
    features = np.array([[
        credit_score,     # CreditScore
        gender_encoded,   # Gender
        geo_germany,      # Geography_Germany
        geo_spain,        # Geography_Spain
        age,              # Age
        tenure,           # Tenure
        balance,          # Balance
        products,         # NumOfProducts
        credit_card,      # HasCrCard
        active_member,    # IsActiveMember
        salary            # EstimatedSalary
    ]])

    # Debug: check raw features 
    st.write("Features passed to model:", features)

    # Load scaler and transform
    scaler = pickle.load(open("scaler.pkl","rb"))
    features_scaled = scaler.transform(features)

    # Debug: check scaled features
    st.write("Scaled features:", features_scaled)

    # Predict
    if st.button("Predict Churn"):
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            st.error("❌ Customer is likely to leave the bank")
        else:
            st.success("✅ Customer will stay with the bank")